/**
 * part of LukasNalbach/compact-texts
 *
 * MIT License
 *
 * Copyright (c) Lukas Nalbach
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>
#include <random>
#include <sstream>
#include <type_traits>
#include <vector>

#include <omp.h>

#include <compact_texts/direct_text.hpp>
#include <compact_texts/packed_text.hpp>
#include <compact_texts/split_text.hpp>

#include <test-fuzz.hpp>
#include <test-inputs.hpp>

using namespace compact_texts;

// symbols that follow a text, such that word reads at its end stay in the allocation
static constexpr uint64_t text_padding = 64;

// maximum length of the random texts of the access and of the comparison tests
static constexpr uint64_t max_access_size = 200000;
static constexpr uint64_t max_compare_size = 50000;
static constexpr uint64_t max_int_access_size = 100000;

// number of random queries per text
static constexpr uint64_t num_order_queries = 1000;
static constexpr uint64_t num_lce_queries = 300;
static constexpr uint64_t num_extract_queries = 100;

// largest penalty of a split text tried by the tests
static constexpr uint64_t max_penalty = 16;

// largest alphabet of the random integer texts
static constexpr uint64_t max_int_sigma = uint64_t{1} << 18;

thread_local std::mt19937_64 gen(test_seed());

template <typename val_t>
static val_t uniform(val_t min, val_t max) {
  return std::uniform_int_distribution<val_t>(min, max)(gen);
}

static uint16_t random_num_threads() { return uint16_t(uniform<int>(1, omp_get_max_threads())); }

static bool coin() { return uniform<int>(0, 1) == 0; }

static std::vector<uint8_t> random_text(uint64_t max_size, uint64_t sigma) {
  const uint64_t low = uniform<uint64_t>(0, 256 - sigma);
  std::vector<uint8_t> text = random_repetitive_input<std::vector<uint8_t>>(
      1, max_size, uint8_t(low), uint8_t(low + sigma - 1), gen());
  text.reserve(text.size() + text_padding);
  return text;
}

static std::vector<uint8_t> random_text(uint64_t max_size) {
  return random_text(max_size, random_log_uniform_size(1, 256, gen));
}

// a random text in which some positions hold symbols from a second, rare range
static std::vector<uint8_t> random_skewed_text(uint64_t max_size) {
  std::vector<uint8_t> text = random_text(max_size);
  std::uniform_real_distribution<double> prob(0.0, 1.0);
  const double rare_share = prob(gen) / 10;
  const uint64_t rare_sigma = random_log_uniform_size(1, 256, gen);
  const uint64_t rare_low = uniform<uint64_t>(0, 256 - rare_sigma);

  for (uint8_t& c : text) {
    if (prob(gen) < rare_share) c = uint8_t(uniform<uint64_t>(rare_low, rare_low + rare_sigma - 1));
  }

  return text;
}

static std::vector<uint8_t> random_maybe_skewed_text(uint64_t max_size) {
  return coin() ? random_text(max_size) : random_skewed_text(max_size);
}

static const char* chars(const std::vector<uint8_t>& t) { return reinterpret_cast<const char*>(t.data()); }

static uint64_t num_distinct(const std::vector<uint8_t>& t) {
  std::vector<bool> used(256, false);
  for (uint8_t c : t) used[c] = true;
  return uint64_t(std::count(used.begin(), used.end(), true));
}

template <typename sym_t>
static uint64_t naive_lce(const std::vector<sym_t>& t, uint64_t i, uint64_t j, uint64_t max) {
  uint64_t k = 0;
  while (k < max && std::max(i, j) + k < t.size() && t[i + k] == t[j + k]) ++k;
  return k;
}

template <typename sym_t>
static uint64_t naive_lce_left(const std::vector<sym_t>& t, uint64_t i, uint64_t j, uint64_t max) {
  uint64_t k = 0;
  while (k < max && k <= std::min(i, j) && t[i - k] == t[j - k]) ++k;
  return k;
}

// checks operator[], char_at, to_char, less_char and the cursor against t
template <typename text_t, typename sym_t>
static void check_access(const text_t& text, const std::vector<sym_t>& t) {
  const uint64_t n = t.size();
  ASSERT_EQ(text.size(), n);

  for (uint64_t i = 0; i < n; ++i) {
    ASSERT_EQ(uint64_t(text.to_char(text[i])), uint64_t(t[i])) << "i=" << i << " n=" << n;
    ASSERT_EQ(uint64_t(sym_t(text.char_at(i))), uint64_t(t[i])) << "i=" << i << " n=" << n;
  }

  std::uniform_int_distribution<uint64_t> pos(0, n - 1);

  for (uint64_t q = 0; q < num_order_queries; ++q) {
    const uint64_t i = pos(gen);
    const uint64_t j = pos(gen);
    EXPECT_EQ(text[i] == text[j], t[i] == t[j]);
    EXPECT_EQ(text.less_char(i, j), text[i] < text[j]);

    // the symbols of an integer split text are ordered by frequency, the others by value
    if constexpr (!std::is_same_v<text_t, split_text<uint32_t>>) {
      EXPECT_EQ(text[i] < text[j], t[i] < t[j]);
    }
  }

  const uint64_t start = pos(gen);
  auto cursor = text.cursor_at(start);
  for (uint64_t i = start; i < n; ++i) ASSERT_EQ(cursor.next(), text[i]) << "cursor at " << i;
}

// checks lce, lce_left, equal and hash against t
template <typename text_t, typename sym_t>
static void check_compare(const text_t& text, const std::vector<sym_t>& t) {
  const uint64_t n = t.size();
  std::uniform_int_distribution<uint64_t> pos(0, n - 1);
  std::uniform_int_distribution<uint64_t> max_distrib(0, n + 1);

  for (uint64_t q = 0; q < num_lce_queries; ++q) {
    uint64_t i = pos(gen);
    uint64_t j = pos(gen);
    if (coin() && n > 1) j = (i + random_log_uniform_size(1, n - 1, gen)) % n;
    const uint64_t max = coin() ? std::numeric_limits<uint64_t>::max() : max_distrib(gen);
    EXPECT_EQ(text.lce(i, j, max), naive_lce(t, i, j, max)) << "lce(" << i << "," << j << "," << max << ") n=" << n;
    EXPECT_EQ(text.lce_left(i, j, max), naive_lce_left(t, i, j, max))
        << "lce_left(" << i << "," << j << "," << max << ") n=" << n;

    const uint64_t len = std::min<uint64_t>(n - std::max(i, j), random_log_uniform_size(1, n, gen));
    const bool same = naive_lce(t, i, j, len) == len;
    EXPECT_EQ(text.equal(i, j, len), same) << "equal(" << i << "," << j << "," << len << ") n=" << n;

    if (same) {
      EXPECT_EQ(text.hash(i, len), text.hash(j, len)) << "hash(" << i << "," << j << "," << len << ")";
    }
  }
}

// checks extract against t
template <typename text_t, typename sym_t>
static void check_extract(const text_t& text, const std::vector<sym_t>& t) {
  const uint64_t n = t.size();
  std::uniform_int_distribution<uint64_t> pos(0, n - 1);
  std::vector<sym_t> out;

  for (uint64_t q = 0; q < num_extract_queries; ++q) {
    const uint64_t i = pos(gen);
    const uint64_t len = random_log_uniform_size(1, n - i, gen);
    out.assign(len, 0);
    text.extract(i, len, out.data());

    // a byte text extracts characters, an integer text symbols
    for (uint64_t k = 0; k < len; ++k) {
      const uint64_t c = std::is_same_v<sym_t, uint8_t> ? uint64_t(out[k]) : uint64_t(text.to_char(out[k]));
      ASSERT_EQ(c, uint64_t(t[i + k])) << "extract(" << i << "," << len << ") n=" << n << " k=" << k;
    }
  }
}

// builds a split text of a byte text, either at once or from blocks packed in random order
static split_text<> build_split(const std::vector<uint8_t>& t, uint16_t threads) {
  const uint64_t n = t.size();
  if (uniform<int>(0, 3) == 0) return split_text<>(chars(t), n, threads);
  split_text<>::histogram_t histogram{};
  packed_text<>::add_char_counts(histogram, chars(t), n, threads);
  split_text<> text(histogram, n, uniform<uint64_t>(0, max_penalty));
  const uint64_t block = split_text<>::chunk_symbols *
                         random_log_uniform_size(1, n / split_text<>::chunk_symbols + 1, gen);
  std::vector<uint64_t> starts;
  for (uint64_t at = 0; at < n; at += block) starts.push_back(at);
  std::shuffle(starts.begin(), starts.end(), gen);

#pragma omp parallel for num_threads(threads) schedule(dynamic, 1)
  for (uint64_t k = 0; k < starts.size(); ++k) {
    text.pack(chars(t) + starts[k], std::min(block, n - starts[k]), starts[k], 1);
  }

  text.finish(threads);
  return text;
}

// returns a copy of a text written with serialize() and read with load()
template <typename text_t>
static text_t serialized_copy(const text_t& text) {
  std::stringstream stream;
  text.serialize(stream);
  text_t copy;
  copy.load(stream);
  EXPECT_TRUE(stream.good());
  EXPECT_EQ(copy.size(), text.size());
  EXPECT_EQ(copy.sigma(), text.sigma());
  EXPECT_EQ(copy.size_in_bytes(), text.size_in_bytes());
  return copy;
}

static void check_packed(const std::vector<uint8_t>& t) {
  const packed_text<> p(chars(t), t.size(), random_num_threads());
  const uint64_t sigma = num_distinct(t);
  EXPECT_EQ(p.sigma(), sigma);
  EXPECT_EQ(p.width(), std::max<int>(1, std::bit_width(sigma - 1)));
  check_access(p, t);
  check_compare(p, t);
  const packed_text<> loaded = serialized_copy(p);
  check_access(loaded, t);
  check_compare(loaded, t);
}

static void check_split(const std::vector<uint8_t>& t) {
  const split_text<> s = build_split(t, random_num_threads());
  EXPECT_EQ(s.sigma(), num_distinct(t));
  const auto histogram = packed_text<>::count_chars(chars(t), t.size(), 1);
  EXPECT_EQ(split_text<>::size_in_bytes_for(histogram, t.size(), s.penalty()), s.size_in_bytes());
  check_access(s, t);
  check_compare(s, t);
  check_extract(s, t);
  const split_text<> loaded = serialized_copy(s);
  check_access(loaded, t);
  check_compare(loaded, t);
  check_extract(loaded, t);
}

static void check_direct(std::vector<uint8_t> t) {
  const direct_text<uint8_t> d(t.data(), t.size());
  EXPECT_EQ(d.sigma(), uint64_t{256});
  check_access(d, t);
  check_compare(d, t);
}

static void test_reverse() {
  std::vector<uint8_t> t = random_text(max_access_size);
  const uint16_t threads = random_num_threads();
  packed_text<> p(chars(t), t.size(), threads);
  std::vector<uint8_t> r = t;
  direct_text<uint8_t> d(r.data(), r.size());
  p.reverse(threads);
  d.reverse(threads);
  std::reverse(t.begin(), t.end());
  ASSERT_EQ(r, t);
  for (uint64_t i = 0; i < t.size(); ++i) ASSERT_EQ(p.to_char(p[i]), t[i]) << "i=" << i;
}

static std::vector<uint32_t> random_int_text(uint64_t max_size, uint64_t& sigma) {
  sigma = random_log_uniform_size(1, max_int_sigma, gen);
  std::vector<uint32_t> text = random_repetitive_input<std::vector<uint32_t>>(
      1, max_size, uint32_t(0), uint32_t(sigma - 1), gen());
  std::uniform_real_distribution<double> prob(0.0, 1.0);

  // makes a few symbols frequent
  if (coin()) {
    const double frequent_share = prob(gen);
    const uint64_t frequent = std::min<uint64_t>(sigma, random_log_uniform_size(1, text_padding, gen));

    for (uint32_t& c : text) {
      if (prob(gen) < frequent_share) c = uint32_t(uniform<uint64_t>(0, frequent - 1));
    }
  }

  text.reserve(text.size() + text_padding);
  return text;
}

// builds a packed or split integer text, either at once or from blocks packed in random order
template <typename text_t>
static text_t build_int_text(std::vector<uint32_t>& t, uint64_t sigma, uint16_t threads) {
  const uint64_t n = t.size();
  if (uniform<int>(0, 3) == 0) return text_t(t.data(), n, sigma, threads);
  typename text_t::histogram_t histogram(sigma, 0);
  for (uint32_t c : t) ++histogram[c];

  text_t text = [&]() {
    if constexpr (std::is_same_v<text_t, split_text<uint32_t>>) {
      return text_t(histogram, n, uniform<uint64_t>(0, max_penalty));
    } else {
      return text_t(histogram, n);
    }
  }();

  const uint64_t block = text_t::chunk_symbols * random_log_uniform_size(1, n / text_t::chunk_symbols + 1, gen);
  std::vector<uint64_t> starts;
  for (uint64_t at = 0; at < n; at += block) starts.push_back(at);
  std::shuffle(starts.begin(), starts.end(), gen);

#pragma omp parallel for num_threads(threads) schedule(dynamic, 1)
  for (uint64_t k = 0; k < starts.size(); ++k) {
    text.pack(t.data() + starts[k], std::min(block, n - starts[k]), starts[k], 1);
  }

  if constexpr (requires { text.finish(threads); }) text.finish(threads);
  return text;
}

// runs fnc(text, reference, sigma) on t as a direct text of packed symbols of t_bytes bytes, each symbol multiplied
// by a random factor (the reference has the same symbols in 64 bits)
template <uint8_t t_bytes, typename fnc_t>
static void with_packed_int_text(const std::vector<uint32_t>& t, uint64_t sigma, fnc_t fnc) {
  using char_t = packed_unsigned<t_bytes>;
  // the symbols are reduced to the alphabet of the narrowest packed type
  const uint64_t base_sigma = std::min<uint64_t>(sigma, uint64_t{1} << (8 * sizeof(packed_unsigned<3>)));
  const uint64_t mult = random_log_uniform_size(1, char_t::max / base_sigma, gen);
  std::vector<uint64_t> t64(t.size());
  std::vector<char_t> packed(t.size());

  for (uint64_t i = 0; i < t.size(); ++i) {
    t64[i] = uint64_t(t[i] % base_sigma) * mult;
    packed[i] = t64[i];
  }

  packed.reserve(packed.size() + text_padding);
  const uint64_t packed_sigma = (base_sigma - 1) * mult + 1;
  fnc(direct_text<char_t>(packed.data(), packed.size(), packed_sigma), t64, packed_sigma);
}

// runs fnc(text, reference, sigma) on an integer text in a random encoding
template <typename fnc_t>
static void with_random_int_text(std::vector<uint32_t>& t, uint64_t sigma, fnc_t fnc) {
  const uint16_t threads = random_num_threads();

  switch (uniform<int>(0, 8)) {
    case 0: fnc(direct_text<uint32_t>(t.data(), t.size(), sigma), t, sigma); break;
    case 1: fnc(build_int_text<packed_text<uint32_t>>(t, sigma, threads), t, sigma); break;
    case 2: fnc(build_int_text<split_text<uint32_t>>(t, sigma, threads), t, sigma); break;
    case 3: {
      std::vector<uint16_t> t16(t.begin(), t.end());
      const uint64_t sigma16 = std::min<uint64_t>(sigma, uint64_t{1} << 16);
      for (uint16_t& c : t16) c = uint16_t(c % sigma16);
      fnc(direct_text<uint16_t>(t16.data(), t16.size(), sigma16), t16, sigma16);
      break;
    }
    case 4: {
      const uint64_t mult = random_log_uniform_size(1, uint40_t::max / sigma, gen);
      std::vector<uint64_t> t64(t.size());
      for (uint64_t i = 0; i < t.size(); ++i) t64[i] = uint64_t(t[i]) * mult;
      const uint64_t sigma64 = (sigma - 1) * mult + 1;
      fnc(direct_text<uint64_t>(t64.data(), t64.size(), sigma64), t64, sigma64);
      break;
    }
    case 5: with_packed_int_text<3>(t, sigma, fnc); break;
    case 6: with_packed_int_text<5>(t, sigma, fnc); break;
    case 7: with_packed_int_text<6>(t, sigma, fnc); break;
    default: with_packed_int_text<7>(t, sigma, fnc); break;
  }
}

static void test_int_access() {
  uint64_t sigma = 0;
  std::vector<uint32_t> t = random_int_text(max_int_access_size, sigma);

  with_random_int_text(t, sigma, [&](const auto& s, const auto& ref, uint64_t ref_sigma) {
    using text_t = std::decay_t<decltype(s)>;
    EXPECT_EQ(s.sigma(), ref_sigma);

    if constexpr (std::is_same_v<text_t, split_text<uint32_t>>) {
      typename text_t::histogram_t histogram(sigma, 0);
      for (uint32_t c : t) ++histogram[c];
      EXPECT_EQ(text_t::size_in_bytes_for(histogram, t.size(), s.penalty()), s.size_in_bytes());
      check_extract(s, ref);
    }

    if constexpr (std::is_same_v<text_t, packed_text<uint32_t>>) check_access(serialized_copy(s), ref);
    check_access(s, ref);
  });
}

static void test_int_compare() {
  uint64_t sigma = 0;
  std::vector<uint32_t> t = random_int_text(max_compare_size, sigma);
  with_random_int_text(t, sigma, [&](const auto& s, const auto& ref, uint64_t) { check_compare(s, ref); });
}

// texts of length 0
static void test_empty() {
  std::vector<uint8_t> t;
  t.reserve(text_padding);
  const packed_text<> p(chars(t), 0, random_num_threads());
  EXPECT_EQ(p.size(), uint64_t{0});
  EXPECT_EQ(p.sigma(), uint64_t{0});
  const split_text<> s(chars(t), 0, random_num_threads());
  EXPECT_EQ(s.size(), uint64_t{0});
  const split_text<>::histogram_t histogram{};
  EXPECT_EQ(split_text<>::size_in_bytes_for(histogram, 0, s.penalty()), s.size_in_bytes());
  direct_text<uint8_t> d(t.data(), 0);
  EXPECT_EQ(d.size(), uint64_t{0});
  d.reverse(random_num_threads());

  std::vector<uint32_t> ti;
  ti.reserve(text_padding);
  const uint64_t sigma = random_log_uniform_size(1, max_int_sigma, gen);
  EXPECT_EQ(packed_text<uint32_t>(ti.data(), 0, sigma, random_num_threads()).size(), uint64_t{0});
  EXPECT_EQ(split_text<uint32_t>(ti.data(), 0, sigma, random_num_threads()).size(), uint64_t{0});
  EXPECT_EQ(direct_text<uint32_t>(ti.data(), 0, sigma).size(), uint64_t{0});
}

// texts of length 1
static void test_single() {
  std::vector<uint8_t> t(1, uint8_t(uniform<int>(0, 255)));
  t.reserve(text_padding);
  check_direct(t);
  check_packed(t);
  check_split(t);
  EXPECT_EQ(packed_text<>(chars(t), 1).lce(0, 0), uint64_t{1});
  EXPECT_EQ(split_text<>(chars(t), 1).lce_left(0, 0), uint64_t{1});
}

// texts with one distinct symbol
static void test_sigma_1() {
  std::vector<uint8_t> t = random_text(max_access_size, 1);
  check_packed(t);
  check_split(t);
  const packed_text<> p(chars(t), t.size(), random_num_threads());
  EXPECT_EQ(p.lce(0, t.size() - 1), uint64_t{1});
  EXPECT_EQ(p.lce(0, 0), t.size());
}

// texts with all 256 byte values
static void test_sigma_256() {
  std::vector<uint8_t> t = random_maybe_skewed_text(max_access_size);
  std::vector<uint8_t> all(256);
  for (uint64_t c = 0; c < 256; ++c) all[c] = uint8_t(c);
  std::shuffle(all.begin(), all.end(), gen);
  t.insert(t.begin() + uniform<uint64_t>(0, t.size()), all.begin(), all.end());
  t.reserve(t.size() + text_padding);
  ASSERT_EQ(num_distinct(t), uint64_t{256});
  check_direct(t);
  check_packed(t);
  check_split(t);
}

TEST(compact_texts, direct_access_and_compare) {
  run_fuzz("direct_text", {{"bytes", [](uint64_t) { check_direct(random_text(max_access_size)); }}});
}

TEST(compact_texts, packed_access_and_compare) {
  run_fuzz("packed_text", {{"bytes", [](uint64_t) { check_packed(random_text(max_access_size)); }}});
}

TEST(compact_texts, split_access_and_compare) {
  run_fuzz("split_text", {{"bytes", [](uint64_t) { check_split(random_maybe_skewed_text(max_access_size)); }}});
}

TEST(compact_texts, reverse) { run_fuzz("text", {{"reverse", [](uint64_t) { test_reverse(); }}}); }

TEST(compact_texts, integers) {
  run_fuzz("integer_text", {{"access", [](uint64_t) { test_int_access(); }},
                            {"compare", [](uint64_t) { test_int_compare(); }}});
}

TEST(compact_texts, edge_cases) {
  run_fuzz("text", {{"n=0", [](uint64_t) { test_empty(); }},
                    {"n=1", [](uint64_t) { test_single(); }},
                    {"sigma=1", [](uint64_t) { test_sigma_1(); }},
                    {"sigma=256", [](uint64_t) { test_sigma_256(); }}});
}
