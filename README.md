# compact-texts

This repository contains C++20 classes that store a text with random access to its symbols and fast comparisons of
substrings, together with tests.

- `direct_text` works on the caller's array, with one byte or one integer of 1 to 8 bytes per symbol.
- `packed_text` stores ceil(log2 sigma) bits per symbol, for bytes and 32-bit integers.
- `split_text` stores fewer bits for frequent symbols, for bytes and 32-bit integers.

The classes are header-only and in the namespace `compact_texts`. Each of them answers `text[i]`, `char_at(i)`,
`cursor_at(i)` (reads the symbols from position `i` on), `lce(i, j)` and `lce_left(i, j)` (the longest common
extension to the right and to the left), `equal(i, j, len)`, `hash(i, len)`, `size()`, `sigma()` and `size_in_bytes()`.
`packed_text` and `split_text` are built in parallel with OpenMP.

## Contents

- [Build](#build)
- [C++ usage](#c-usage)
- [Tests](#tests)
- [License](#license)

## Build

Requirements: a C++20 compiler, CMake 3.21 or later and OpenMP.

```shell
git clone https://github.com/LukasNalbach/compact-texts.git
cd compact-texts
cmake -B build -G Ninja
cmake --build build
```

`cmake` downloads the submodule [string-utils](https://github.com/LukasNalbach/string-utils), which provides the build
settings and the memory and test helpers. The test is written to `build/tests/` and is built with
`COMPACT_TEXTS_BUILD_TESTS` (`ON` if compact-texts is the top-level project). The other options and the environment
variables are described in string-utils (with the prefix `COMPACT_TEXTS`); with `COMPACT_TEXTS_MARCH_NATIVE` (`ON`),
`split_text` uses AVX2 or AVX-512 VBMI if the CPU supports them.

## C++ usage

Add compact-texts to a CMake project and link the target `compact_texts`:

```cmake
add_subdirectory(external/compact-texts EXCLUDE_FROM_ALL)
target_link_libraries(<target> PRIVATE compact_texts)
```

If the project uses string-utils itself, it adds string-utils first; compact-texts then uses that copy.

```cpp
#include <iostream>
#include <string>

#include <compact_texts/direct_text.hpp>
#include <compact_texts/packed_text.hpp>
#include <compact_texts/split_text.hpp>

int main() {
    std::string input = "abracadabra abracadabra";
    const uint64_t n = input.size();

    compact_texts::direct_text<char> direct(input.data(), n);
    compact_texts::packed_text<> packed(input.data(), n);
    compact_texts::split_text<> split(input.data(), n);

    std::cout << "sigma: " << packed.sigma() << ", bits per symbol: "
              << int(packed.width()) << std::endl; // 6, 3
    std::cout << packed.char_at(4) << split.char_at(5) << std::endl; // ca

    // longest common extension of the suffixes at 0 and 12
    std::cout << direct.lce(0, 12) << " " << packed.lce(0, 12) << " "
              << split.lce(0, 12) << std::endl; // 11 11 11

    // longest common extension of the prefixes ending at 10 and 22
    std::cout << split.lce_left(10, 22) << std::endl; // 11

    std::cout << std::boolalpha << packed.equal(0, 7, 4) << std::endl; // true

    auto cursor = split.cursor_at(1);
    for (int k = 0; k < 3; k++) std::cout << split.to_char(cursor.next());
    std::cout << std::endl; // bra
}
```

## Tests

```shell
build/tests/test-compact-texts
```

The tests check random inputs: at most `COMPACT_TEXTS_TEST_SECONDS` seconds (default 10) or
`COMPACT_TEXTS_TEST_ITERATIONS` inputs (default 1000) per function. A failed test prints its seed;
`COMPACT_TEXTS_TEST_SEED=<seed>` repeats its inputs.

## License

This project is licensed under the MIT License, see [LICENSE](LICENSE). string-utils and the libraries in its
`external/` have their own licenses.
