# compact-texts

This repository contains C++20 classes that store a text compactly with random access to its symbols and word-wise access for fast comparisons: `direct_text` with one byte (or one integer) per symbol, `packed_text` with ceil(log2 sigma) bits per symbol, and `split_text` with variable-width symbols.

## Contents

All classes are in the namespace `compact_texts` and in [include/compact_texts/](include/compact_texts/). Each of them stores a byte text (`char`) or a text of integers, and answers for positions `i` and `j`:

- `text[i]`: the symbol at `i` (its rank among the occurring symbols for packed and split byte texts), `text.char_at(i)`: the character at `i`, `text.to_char(symbol)`: the character of a symbol, `text.less_char(i, j)`: whether `text[i] < text[j]`
- `text.cursor_at(i)`: a cursor whose `next()` returns the symbols at `i`, `i + 1`, ...
- `text.lce(i, j, max)` and `text.lce_left(i, j, max)`: the length (at most `max`) of the longest common extension of the suffixes starting at `i` and `j` and of the prefixes ending at `i` and `j`, compared 64 bits at a time
- `text.equal(i, j, len)`: whether `text[i..i+len)` and `text[j..j+len)` are equal, `text.hash(i, len)`: a hash value of `text[i..i+len)`
- `text.size()`, `text.sigma()` and `text.size_in_bytes()`

| Class | Symbol types | Bits per symbol |
| --- | --- | --- |
| `direct_text<t>` | `char`, `uint8_t`, 16-, 32-, 40- (`uint40_t`) and 64-bit integers | `8 * sizeof(t)`; the text stays in the caller's array |
| `packed_text<>`, `packed_text<uint32_t>` | bytes, 32-bit integers | ceil(log2 sigma) |
| `split_text<>`, `split_text<uint32_t>` | bytes, 32-bit integers | the symbols are grouped by frequency into classes; each position stores the code of its class with a fixed number of bits and, unless its class has only one symbol, the position of the symbol within its class with 1 to 8 more bits |

A byte text in `packed_text` and `split_text` maps the occurring characters to the symbols `0, ..., sigma - 1` in the order of the characters; an integer text in `split_text` maps its values to symbols in the order of their frequency. `packed_text` and `split_text` can also be built in parts: construct them from the histogram of the text and its length, call `pack(data, length, at)` for each part (also from several threads for disjoint parts that start at multiples of `chunk_symbols`), then (only `split_text`) `finish()`. `split_text` also writes ranges to an array (`extract(i, len, out)`: the characters of a byte text, the symbols of an integer text) and gives its size for a histogram before building it (`size_in_bytes_for`). `direct_text` and `packed_text` can reverse the text (`reverse()`). `packed_text` and the byte text in `split_text` write themselves as they are in memory to a stream (`serialize(out)`) and read such a stream (`load(in)`).

The construction of `packed_text` and `split_text` and `reverse()` run in parallel with OpenMP (the last argument is the number of threads).

## Build

```shell
git clone https://github.com/LukasNalbach/compact-texts.git
cd compact-texts
cmake -B build -G Ninja
cmake --build build
build/tests/test-compact-texts
```

compact-texts uses [string-utils](https://github.com/LukasNalbach/string-utils) (submodule `external/string-utils`) for its build settings, memory helpers and test helpers. The build creates the test `test-compact-texts`; the classes are header-only.

## Usage in CMake

A project adds compact-texts as a submodule under `external/compact-texts` and links the target `compact_texts`:

```cmake
add_subdirectory(external/compact-texts)
target_link_libraries(<target> PRIVATE compact_texts)
```

If the project uses string-utils itself, it adds string-utils first; compact-texts then uses that copy.

## Example

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

## Options and environment variables

The options and environment variables of string-utils carry the prefix `COMPACT_TEXTS` when compact-texts is the top-level project (see the README of string-utils):

| Option | Default | Description |
| --- | --- | --- |
| `COMPACT_TEXTS_BUILD_TESTS` | `ON` if top-level | build the tests |
| `COMPACT_TEXTS_MARCH_NATIVE` | `ON` | compile for the host CPU (`-march=native`); `split_text` uses AVX2 or AVX-512 VBMI if available |
| `COMPACT_TEXTS_FAST_TESTS` | `OFF` | compile the tests with `-O0` |

| Environment variable | Description |
| --- | --- |
| `COMPACT_TEXTS_HUGE_PAGES` | `0`: no advice, `< 0`: no huge pages, else (default): huge pages for large allocations |
| `COMPACT_TEXTS_TEST_SECONDS` | maximum time in seconds per tested functionality (default 10) |
| `COMPACT_TEXTS_TEST_ITERATIONS` | maximum number of random inputs per tested functionality (default 1000) |
| `COMPACT_TEXTS_TEST_SEED` | replays the random inputs of a seed that a failed test printed |

## License

This project is licensed under the MIT License, see [LICENSE](LICENSE). string-utils and the libraries in its `external/` keep their own licenses.
