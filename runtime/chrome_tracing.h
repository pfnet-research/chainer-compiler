#pragma once

#include <stdint.h>

#include <chrono>
#include <memory>
#include <string>
#include <vector>

namespace chainer_compiler {
namespace runtime {

class ChromeTracingEmitter {
public:
    struct Event {
        Event(const std::string& c, const std::string& n, int p, int64_t f);
        void Finish();
        std::string category;
        std::string name;
        int pc;
        int64_t flops;
        std::chrono::system_clock::time_point start_time;
        std::chrono::system_clock::time_point end_time;
    };

    class ScopedEvent {
    public:
        explicit ScopedEvent(
                ChromeTracingEmitter* chrome_tracing, const std::string& category, const std::string& name, int pc = -1, int64_t flops = 0);
        ~ScopedEvent();

    private:
        Event* event_;
    };

    ChromeTracingEmitter();

    // Takes the ownership of `event`.
    void AddEvent(Event* event);

    void Emit(const std::string& output_filename) const;

private:
    std::vector<std::unique_ptr<Event>> events_;
    std::chrono::system_clock::time_point base_time_;
};

}  // namespace runtime
}  // namespace chainer_compiler
