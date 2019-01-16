#include "chrome_tracing.h"

#include <cstdint>
#include <fstream>

namespace chainer_compiler {
namespace runtime {

ChromeTracingEmitter::ChromeTracingEmitter() : base_time_(std::chrono::system_clock::now()) {
}

void ChromeTracingEmitter::AddEvent(Event* event) {
    events_.emplace_back(event);
}

ChromeTracingEmitter::Event::Event(const std::string& c, const std::string& n, int p)
    : category(c), name(n), pc(p), start_time(std::chrono::system_clock::now()) {
}

void ChromeTracingEmitter::Event::Finish() {
    end_time = std::chrono::system_clock::now();
}

ChromeTracingEmitter::ScopedEvent::ScopedEvent(
        ChromeTracingEmitter* chrome_tracing, const std::string& category, const std::string& name, int pc)
    : event_(nullptr) {
    if (chrome_tracing) {
        event_ = new ChromeTracingEmitter::Event(category, name, pc);
        chrome_tracing->AddEvent(event_);
    }
}

ChromeTracingEmitter::ScopedEvent::~ScopedEvent() {
    if (event_) {
        event_->Finish();
    }
}

void ChromeTracingEmitter::Emit(const std::string& output_filename) const {
    std::ofstream ofs(output_filename);
    ofs << "[\n";
    bool is_first = true;
    for (const std::unique_ptr<Event>& event : events_) {
        int64_t ts = std::chrono::duration_cast<std::chrono::microseconds>(event->start_time - base_time_).count();
        int64_t dur = std::chrono::duration_cast<std::chrono::microseconds>(event->end_time - event->start_time).count();

        if (!is_first) {
            ofs << ",\n";
        }
        is_first = false;
        ofs << "{";
        ofs << "\"cat\":\"" << event->category << "\",";
        ofs << "\"name\":\"" << event->name << "\",";
        ofs << "\"ts\":" << ts << ",";
        ofs << "\"dur\":" << dur << ",";
        ofs << "\"tid\":1,";
        ofs << "\"pid\":1,";
        if (event->pc >= 0) {
            ofs << "\"args\":{\"pc\":" << event->pc << "},";
        }
        ofs << "\"ph\":\"X\"";
        ofs << "}";
    }
    ofs << "]\n";
}

}  // namespace runtime
}  // namespace chainer_compiler
