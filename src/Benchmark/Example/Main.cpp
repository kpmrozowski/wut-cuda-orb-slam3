#include <orb/Benchmark.h>

int foo(int value) {
    if (value < 1)
        return 0;
    if (value == 1)
        return 1;
    return foo(value - 1) + foo(value - 2);
}

int bar(int value) {
    int i = 0, k = 1;
    for  (int j = 0; j < value; ++j) {
        auto tmp = k;
        k += i;
        i = tmp;
    }
    return i;
}

int main() {
    orb::benchmark::Benchmark::the().start("out.csv");

    for (auto i = 0; i < 40; ++i) {
        auto foo_res = orb::benchmark::measure_function(orb::benchmark::MeasuredFunction::Foo, foo, i);
        auto bar_res = orb::benchmark::measure_function(orb::benchmark::MeasuredFunction::Bar, bar, i);
        if (foo_res != bar_res) {
            orb::benchmark::Benchmark::the().stop();
            return 1;
        }
    }

    return static_cast<int>(orb::benchmark::Benchmark::the().stop());
}