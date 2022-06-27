#include "orb/Benchmark.h"
#include <fstream>

namespace orb::benchmark {

Benchmark &Benchmark::the() {
    static Benchmark instance;
    return instance;
}

ExitStatus Benchmark::write_routine(const std::string &filename) {
    std::ofstream out_file(filename);
    if (not out_file.is_open())
        return ExitStatus::CouldNotOpenFile;

    out_file << "function_name,duration\n";

    for (;;) {
        auto entry = m_log_queue.try_pop();
        if (not entry.has_value()) {
            m_log_queue.swap_queues();
            continue;
        }

        if (entry->function == MeasuredFunction::Action_Exit)
            return ExitStatus::Ok;

        out_file << to_string(entry->function) << "," << entry->duration.count() << '\n';
    }
}

ExitStatus Benchmark::stop() {
    m_log_queue.emplace(MeasuredFunction::Action_Exit, Duration{});
    return m_worker_thread.get();
}

void Benchmark::log(MeasuredFunction func, Duration duration) {
    m_log_queue.emplace(func, duration);
}

void Benchmark::start(const std::string &filename) {
    m_worker_thread = std::async(std::launch::async, &Benchmark::write_routine, this, filename);
}

}
