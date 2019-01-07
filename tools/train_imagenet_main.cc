#include <tools/train_imagenet.h>

int main(int argc, char** argv) {
    std::vector<std::string> args;
    for (int i = 0; i < argc; ++i) {
        args.push_back(argv[i]);
    }
    oniku::runtime::TrainImagenet(args);
}
