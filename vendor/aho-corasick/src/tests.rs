use std::collections::HashMap;
use std::io;
use std::usize;

use {AhoCorasickBuilder, Match, MatchKind};

/// A description of a single test against an Aho-Corasick automaton.
///
/// A single test may not necessarily pass on every configuration of an
/// Aho-Corasick automaton. The tests are categorized and grouped appropriately
/// below.
#[derive(Clone, Debug, Eq, PartialEq)]
struct SearchTest {
    /// The name of this test, for debugging.
    name: &'static str,
    /// The patterns to search for.
    patterns: &'static [&'static str],
    /// The text to search.
    haystack: &'static str,
    /// Each match is a triple of (pattern_index, start, end), where
    /// pattern_index is an index into `patterns` and `start`/`end` are indices
    /// into `haystack`.
    matches: &'static [(usize, usize, usize)],
}

/// Short-hand constructor for SearchTest. We use it a lot below.
macro_rules! t {
    ($name:ident, $patterns:expr, $haystack:expr, $matches:expr) => {
        SearchTest {
            name: stringify!($name),
            patterns: $patterns,
            haystack: $haystack,
            matches: $matches,
        }
    }
}

/// A collection of test groups.
type TestCollection = &'static [&'static [SearchTest]];

// Define several collections corresponding to the different type of match
// semantics supported by Aho-Corasick. These collections have some overlap,
// but each collection should have some tests that no other collection has.

/// Tests for Aho-Corasick's standard non-overlapping match semantics.
const AC_STANDARD_NON_OVERLAPPING: TestCollection = &[
    BASICS, NON_OVERLAPPING, STANDARD, REGRESSION,
];

/// Tests for Aho-Corasick's standard overlapping match semantics.
const AC_STANDARD_OVERLAPPING: TestCollection = &[
    BASICS, OVERLAPPING, REGRESSION,
];

/// Tests for Aho-Corasick's leftmost-first match semantics.
const AC_LEFTMOST_FIRST: TestCollection = &[
    BASICS, NON_OVERLAPPING, LEFTMOST, LEFTMOST_FIRST, REGRESSION,
];

/// Tests for Aho-Corasick's leftmost-longest match semantics.
const AC_LEFTMOST_LONGEST: TestCollection = &[
    BASICS, NON_OVERLAPPING, LEFTMOST, LEFTMOST_LONGEST, REGRESSION,
];

// Now define the individual tests that make up the collections above.

/// A collection of tests for the Aho-Corasick algorithm that should always be
/// true regardless of match semantics. That is, all combinations of
/// leftmost-{shortest, first, longest} x {overlapping, non-overlapping}
/// should produce the same answer.
const BASICS: &'static [SearchTest] = &[
    t!(basic000, &["a"], "", &[]),
    t!(basic010, &["a"], "a", &[(0, 0, 1)]),
    t!(basic020, &["a"], "aa", &[(0, 0, 1), (0, 1, 2)]),
    t!(basic030, &["a"], "aaa", &[(0, 0, 1), (0, 1, 2), (0, 2, 3)]),
    t!(basic040, &["a"], "aba", &[(0, 0, 1), (0, 2, 3)]),
    t!(basic050, &["a"], "bba", &[(0, 2, 3)]),
    t!(basic060, &["a"], "bbb", &[]),
    t!(basic070, &["a"], "bababbbba", &[(0, 1, 2), (0, 3, 4), (0, 8, 9)]),

    t!(basic100, &["aa"], "", &[]),
    t!(basic110, &["aa"], "aa", &[(0, 0, 2)]),
    t!(basic120, &["aa"], "aabbaa", &[(0, 0, 2), (0, 4, 6)]),
    t!(basic130, &["aa"], "abbab", &[]),
    t!(basic140, &["aa"], "abbabaa", &[(0, 5, 7)]),

    t!(basic200, &["abc"], "abc", &[(0, 0, 3)]),
    t!(basic210, &["abc"], "zazabzabcz", &[(0, 6, 9)]),
    t!(basic220, &["abc"], "zazabczabcz", &[(0, 3, 6), (0, 7, 10)]),

    t!(basic300, &["a", "b"], "", &[]),
    t!(basic310, &["a", "b"], "z", &[]),
    t!(basic320, &["a", "b"], "b", &[(1, 0, 1)]),
    t!(basic330, &["a", "b"], "a", &[(0, 0, 1)]),
    t!(basic340, &["a", "b"], "abba", &[
       (0, 0, 1), (1, 1, 2), (1, 2, 3), (0, 3, 4),
    ]),
    t!(basic350, &["b", "a"], "abba", &[
       (1, 0, 1), (0, 1, 2), (0, 2, 3), (1, 3, 4),
    ]),
    t!(nover360, &["abc", "bc"], "xbc", &[
       (1, 1, 3),
    ]),

    t!(basic400, &["foo", "bar"], "", &[]),
    t!(basic410, &["foo", "bar"], "foobar", &[
       (0, 0, 3), (1, 3, 6),
    ]),
    t!(basic420, &["foo", "bar"], "barfoo", &[
       (1, 0, 3), (0, 3, 6),
    ]),
    t!(basic430, &["foo", "bar"], "foofoo", &[
       (0, 0, 3), (0, 3, 6),
    ]),
    t!(basic440, &["foo", "bar"], "barbar", &[
       (1, 0, 3), (1, 3, 6),
    ]),
    t!(basic450, &["foo", "bar"], "bafofoo", &[
       (0, 4, 7),
    ]),
    t!(basic460, &["bar", "foo"], "bafofoo", &[
       (1, 4, 7),
    ]),
    t!(basic470, &["foo", "bar"], "fobabar", &[
       (1, 4, 7),
    ]),
    t!(basic480, &["bar", "foo"], "fobabar", &[
       (0, 4, 7),
    ]),

    t!(basic600, &[""], "", &[(0, 0, 0)]),
    t!(basic610, &[""], "a", &[(0, 0, 0), (0, 1, 1)]),
    t!(basic620, &[""], "abc", &[(0, 0, 0), (0, 1, 1), (0, 2, 2), (0, 3, 3)]),

    t!(basic700, &["yabcdef", "abcdezghi"], "yabcdefghi", &[
       (0, 0, 7),
    ]),
    t!(basic710, &["yabcdef", "abcdezghi"], "yabcdezghi", &[
       (1, 1, 10),
    ]),
    t!(basic720, &["yabcdef", "bcdeyabc", "abcdezghi"], "yabcdezghi", &[
       (2, 1, 10),
    ]),
];

/// Tests for non-overlapping standard match semantics.
///
/// These tests generally shouldn't pass for leftmost-{first,longest}, although
/// some do in order to write clearer tests. For example, standard000 will
/// pass with leftmost-first semantics, but standard010 will not. We write
/// both to emphasize how the match semantics work.
const STANDARD: &'static [SearchTest] = &[
    t!(standard000, &["ab", "abcd"], "abcd", &[(0, 0, 2)]),
    t!(standard010, &["abcd", "ab"], "abcd", &[(1, 0, 2)]),
    t!(standard020, &["abcd", "ab", "abc"], "abcd", &[(1, 0, 2)]),
    t!(standard030, &["abcd", "abc", "ab"], "abcd", &[(2, 0, 2)]),
    t!(standard040, &["a", ""], "a", &[(1, 0, 0), (1, 1, 1)]),

    t!(standard400, &["abcd", "bcd", "cd", "b"], "abcd", &[
       (3, 1, 2), (2, 2, 4),
    ]),
    t!(standard410, &["", "a"], "a", &[
       (0, 0, 0), (0, 1, 1),
    ]),
    t!(standard420, &["", "a"], "aa", &[
       (0, 0, 0), (0, 1, 1), (0, 2, 2),
    ]),
    t!(standard430, &["", "a", ""], "a", &[
       (0, 0, 0), (0, 1, 1),
    ]),
    t!(standard440, &["a", "", ""], "a", &[
       (1, 0, 0), (1, 1, 1),
    ]),
    t!(standard450, &["", "", "a"], "a", &[
       (0, 0, 0), (0, 1, 1),
    ]),
];

/// Tests for non-overlapping leftmost match semantics. These should pass for
/// both leftmost-first and leftmost-longest match kinds. Stated differently,
/// among ambiguous matches, the longest match and the match that appeared
/// first when constructing the automaton should always be the same.
const LEFTMOST: &'static [SearchTest] = &[
    t!(leftmost000, &["ab", "ab"], "abcd", &[(0, 0, 2)]),
    t!(leftmost010, &["a", ""], "a", &[(0, 0, 1), (1, 1, 1)]),
    t!(leftmost020, &["", ""], "a", &[(0, 0, 0), (0, 1, 1)]),

    t!(leftmost300, &["abcd", "bce", "b"], "abce", &[(1, 1, 4)]),
    t!(leftmost310, &["abcd", "ce", "bc"], "abce", &[(2, 1, 3)]),
    t!(leftmost320, &["abcd", "bce", "ce", "b"], "abce", &[(1, 1, 4)]),
    t!(leftmost330, &["abcd", "bce", "cz", "bc"], "abcz", &[(3, 1, 3)]),
    t!(leftmost340, &["bce", "cz", "bc"], "bcz", &[(2, 0, 2)]),
    t!(leftmost350, &["abc", "bd", "ab"], "abd", &[(2, 0, 2)]),
    t!(leftmost360, &["abcdefghi", "hz", "abcdefgh"], "abcdefghz", &[
       (2, 0, 8),
    ]),
    t!(leftmost370, &["abcdefghi", "cde", "hz", "abcdefgh"], "abcdefghz", &[
       (3, 0, 8),
    ]),
    t!(leftmost380, &["abcdefghi", "hz", "abcdefgh", "a"], "abcdefghz", &[
       (2, 0, 8),
    ]),
    t!(leftmost390, &["b", "abcdefghi", "hz", "abcdefgh"], "abcdefghz", &[
       (3, 0, 8),
    ]),
    t!(leftmost400, &["h", "abcdefghi", "hz", "abcdefgh"], "abcdefghz", &[
       (3, 0, 8),
    ]),
    t!(leftmost410, &["z", "abcdefghi", "hz", "abcdefgh"], "abcdefghz", &[
       (3, 0, 8), (0, 8, 9),
    ]),
];

/// Tests for non-overlapping leftmost-first match semantics. These tests
/// should generally be specific to leftmost-first, which means they should
/// generally fail under leftmost-longest semantics.
const LEFTMOST_FIRST: &'static [SearchTest] = &[
    t!(leftfirst000, &["ab", "abcd"], "abcd", &[(0, 0, 2)]),
    t!(leftfirst010, &["", "a"], "a", &[(0, 0, 0), (0, 1, 1)]),
    t!(leftfirst011, &["", "a", ""], "a", &[
       (0, 0, 0), (0, 1, 1),
    ]),
    t!(leftfirst012, &["a", "", ""], "a", &[
       (0, 0, 1), (1, 1, 1),
    ]),
    t!(leftfirst013, &["", "", "a"], "a", &[
       (0, 0, 0), (0, 1, 1),
    ]),
    t!(leftfirst020, &["abcd", "ab"], "abcd", &[(0, 0, 4)]),
    t!(leftfirst030, &["ab", "ab"], "abcd", &[(0, 0, 2)]),

    t!(leftlong100, &["abcdefg", "bcde", "bcdef"], "abcdef", &[(1, 1, 5)]),
    t!(leftlong110, &["abcdefg", "bcdef", "bcde"], "abcdef", &[(1, 1, 6)]),

    t!(leftfirst300, &["abcd", "b", "bce"], "abce", &[(1, 1, 2)]),
    t!(leftfirst310, &["abcd", "b", "bce", "ce"], "abce", &[
       (1, 1, 2), (3, 2, 4),
    ]),
    t!(leftfirst320, &["a", "abcdefghi", "hz", "abcdefgh"], "abcdefghz", &[
       (0, 0, 1), (2, 7, 9),
    ]),
    t!(leftfirst330, &["a", "abab"], "abab", &[(0, 0, 1), (0, 2, 3)]),
];

/// Tests for non-overlapping leftmost-longest match semantics. These tests
/// should generally be specific to leftmost-longest, which means they should
/// generally fail under leftmost-first semantics.
const LEFTMOST_LONGEST: &'static [SearchTest] = &[
    t!(leftlong000, &["ab", "abcd"], "abcd", &[(1, 0, 4)]),
    t!(leftlong010, &["abcd", "bcd", "cd", "b"], "abcd", &[
       (0, 0, 4),
    ]),
    t!(leftlong020, &["", "a"], "a", &[
       (1, 0, 1), (0, 1, 1),
    ]),
    t!(leftlong021, &["", "a", ""], "a", &[
       (1, 0, 1), (0, 1, 1),
    ]),
    t!(leftlong022, &["a", "", ""], "a", &[
       (0, 0, 1), (1, 1, 1),
    ]),
    t!(leftlong023, &["", "", "a"], "a", &[
       (2, 0, 1), (0, 1, 1),
    ]),
    t!(leftlong030, &["", "a"], "aa", &[
       (1, 0, 1), (1, 1, 2), (0, 2, 2),
    ]),
    t!(leftlong040, &["a", "ab"], "a", &[(0, 0, 1)]),
    t!(leftlong050, &["a", "ab"], "ab", &[(1, 0, 2)]),
    t!(leftlong060, &["ab", "a"], "a", &[(1, 0, 1)]),
    t!(leftlong070, &["ab", "a"], "ab", &[(0, 0, 2)]),

    t!(leftlong100, &["abcdefg", "bcde", "bcdef"], "abcdef", &[(2, 1, 6)]),
    t!(leftlong110, &["abcdefg", "bcdef", "bcde"], "abcdef", &[(1, 1, 6)]),

    t!(leftlong300, &["abcd", "b", "bce"], "abce", &[(2, 1, 4)]),
    t!(leftlong310, &["a", "abcdefghi", "hz", "abcdefgh"], "abcdefghz", &[
       (3, 0, 8),
    ]),
    t!(leftlong320, &["a", "abab"], "abab", &[(1, 0, 4)]),
    t!(leftlong330, &["abcd", "b", "ce"], "abce", &[
       (1, 1, 2), (2, 2, 4),
    ]),
];

/// Tests for non-overlapping match semantics.
///
/// Generally these tests shouldn't pass when using overlapping semantics.
/// These should pass for both standard and leftmost match semantics.
const NON_OVERLAPPING: &'static [SearchTest] = &[
    t!(nover010, &["abcd", "bcd", "cd"], "abcd", &[
       (0, 0, 4),
    ]),
    t!(nover020, &["bcd", "cd", "abcd"], "abcd", &[
       (2, 0, 4),
    ]),
    t!(nover030, &["abc", "bc"], "zazabcz", &[
       (0, 3, 6),
    ]),

    t!(nover100, &["ab", "ba"], "abababa", &[
       (0, 0, 2), (0, 2, 4), (0, 4, 6),
    ]),

    t!(nover200, &["foo", "foo"], "foobarfoo", &[
       (0, 0, 3), (0, 6, 9),
    ]),

    t!(nover300, &["", ""], "", &[
       (0, 0, 0),
    ]),
    t!(nover310, &["", ""], "a", &[
       (0, 0, 0), (0, 1, 1),
    ]),
];

/// Tests for overlapping match semantics.
///
/// This only supports standard match semantics, since leftmost-{first,longest}
/// do not support overlapping matches.
const OVERLAPPING: &'static [SearchTest] = &[
    t!(over000, &["abcd", "bcd", "cd", "b"], "abcd", &[
       (3, 1, 2), (0, 0, 4), (1, 1, 4), (2, 2, 4),
    ]),
    t!(over010, &["bcd", "cd", "b", "abcd"], "abcd", &[
       (2, 1, 2), (3, 0, 4), (0, 1, 4), (1, 2, 4),
    ]),
    t!(over020, &["abcd", "bcd", "cd"], "abcd", &[
       (0, 0, 4), (1, 1, 4), (2, 2, 4),
    ]),
    t!(over030, &["bcd", "abcd", "cd"], "abcd", &[
       (1, 0, 4), (0, 1, 4), (2, 2, 4),
    ]),
    t!(over040, &["bcd", "cd", "abcd"], "abcd", &[
       (2, 0, 4), (0, 1, 4), (1, 2, 4),
    ]),
    t!(over050, &["abc", "bc"], "zazabcz", &[
       (0, 3, 6), (1, 4, 6),
    ]),

    t!(over100, &["ab", "ba"], "abababa", &[
       (0, 0, 2), (1, 1, 3), (0, 2, 4), (1, 3, 5), (0, 4, 6), (1, 5, 7),
    ]),

    t!(over200, &["foo", "foo"], "foobarfoo", &[
       (0, 0, 3), (1, 0, 3), (0, 6, 9), (1, 6, 9),
    ]),

    t!(over300, &["", ""], "", &[
       (0, 0, 0), (1, 0, 0),
    ]),
    t!(over310, &["", ""], "a", &[
       (0, 0, 0), (1, 0, 0), (0, 1, 1), (1, 1, 1),
    ]),
    t!(over320, &["", "a"], "a", &[
       (0, 0, 0), (1, 0, 1), (0, 1, 1),
    ]),
    t!(over330, &["", "a", ""], "a", &[
       (0, 0, 0), (2, 0, 0), (1, 0, 1), (0, 1, 1), (2, 1, 1),
    ]),
    t!(over340, &["a", "", ""], "a", &[
       (1, 0, 0), (2, 0, 0), (0, 0, 1), (1, 1, 1), (2, 1, 1),
    ]),
    t!(over350, &["", "", "a"], "a", &[
       (0, 0, 0), (1, 0, 0), (2, 0, 1), (0, 1, 1), (1, 1, 1),
    ]),
];

/// Regression tests that are applied to all Aho-Corasick combinations.
///
/// If regression tests are needed for specific match semantics, then add them
/// to the appropriate group above.
const REGRESSION: &'static [SearchTest] = &[
    t!(regression010, &["inf", "ind"], "infind", &[
       (0, 0, 3), (1, 3, 6),
    ]),
    t!(regression020, &["ind", "inf"], "infind", &[
       (1, 0, 3), (0, 3, 6),
    ]),
    t!(regression030, &["libcore/", "libstd/"], "libcore/char/methods.rs", &[
       (0, 0, 8),
    ]),
    t!(regression040, &["libstd/", "libcore/"], "libcore/char/methods.rs", &[
       (1, 0, 8),
    ]),
    t!(regression050, &["\x00\x00\x01", "\x00\x00\x00"], "\x00\x00\x00", &[
       (1, 0, 3),
    ]),
    t!(regression060, &["\x00\x00\x00", "\x00\x00\x01"], "\x00\x00\x00", &[
       (0, 0, 3),
    ]),
];

// Now define a test for each combination of things above that we want to run.
// Since there are a few different combinations for each collection of tests,
// we define a couple of macros to avoid repetition drudgery. The testconfig
// macro constructs the automaton from a given match kind, and runs the search
// tests one-by-one over the given collection. The `with` parameter allows one
// to configure the builder with additional parameters. The testcombo macro
// invokes testconfig in precisely this way: it sets up several tests where
// each one turns a different knob on AhoCorasickBuilder.

macro_rules! testconfig {
    (overlapping, $name:ident, $collection:expr, $kind:ident, $with:expr) => {
        #[test]
        fn $name() {
            run_search_tests($collection, |test| {
                let mut builder = AhoCorasickBuilder::new();
                $with(&mut builder);
                builder
                    .match_kind(MatchKind::$kind)
                    .build(test.patterns)
                    .find_overlapping_iter(test.haystack)
                    .collect()
            });
        }
    };
    (stream, $name:ident, $collection:expr, $kind:ident, $with:expr) => {
        #[test]
        fn $name() {
            run_search_tests($collection, |test| {
                let buf = io::BufReader::with_capacity(
                    1,
                    test.haystack.as_bytes(),
                );
                let mut builder = AhoCorasickBuilder::new();
                $with(&mut builder);
                builder
                    .match_kind(MatchKind::$kind)
                    .build(test.patterns)
                    .stream_find_iter(buf)
                    .map(|result| result.unwrap())
                    .collect()
            });
        }
    };
    ($name:ident, $collection:expr, $kind:ident, $with:expr) => {
        #[test]
        fn $name() {
            run_search_tests($collection, |test| {
                let mut builder = AhoCorasickBuilder::new();
                $with(&mut builder);
                builder
                    .match_kind(MatchKind::$kind)
                    .build(test.patterns)
                    .find_iter(test.haystack)
                    .collect()
            });
        }
    };
}

macro_rules! testcombo {
    ($name:ident, $collection:expr, $kind:ident) => {
        mod $name {
            use super::*;

            testconfig!(nfa_default, $collection, $kind, |_| ());
            testconfig!(nfa_no_prefilter, $collection, $kind,
                |b: &mut AhoCorasickBuilder| {
                    b.prefilter(false);
                });
            testconfig!(nfa_all_sparse, $collection, $kind,
                |b: &mut AhoCorasickBuilder| {
                    b.dense_depth(0);
                });
            testconfig!(nfa_all_dense, $collection, $kind,
                |b: &mut AhoCorasickBuilder| {
                    b.dense_depth(usize::MAX);
                });
            testconfig!(dfa_default, $collection, $kind,
                |b: &mut AhoCorasickBuilder| {
                    b.dfa(true);
                });
            testconfig!(dfa_no_prefilter, $collection, $kind,
                |b: &mut AhoCorasickBuilder| {
                    b.dfa(true).prefilter(false);
                });
            testconfig!(dfa_all_sparse, $collection, $kind,
                |b: &mut AhoCorasickBuilder| {
                    b.dfa(true).dense_depth(0);
                });
            testconfig!(dfa_all_dense, $collection, $kind,
                |b: &mut AhoCorasickBuilder| {
                    b.dfa(true).dense_depth(usize::MAX);
                });
            testconfig!(dfa_no_byte_class, $collection, $kind,
                |b: &mut AhoCorasickBuilder| {
                    b.dfa(true).byte_classes(false);
                });
            testconfig!(dfa_no_premultiply, $collection, $kind,
                |b: &mut AhoCorasickBuilder| {
                    b.dfa(true).premultiply(false);
                });
            testconfig!(dfa_no_byte_class_no_premultiply, $collection, $kind,
                |b: &mut AhoCorasickBuilder| {
                    b.dfa(true).byte_classes(false).premultiply(false);
                });
        }
    };
}

// Write out the combinations.
testcombo!(search_leftmost_longest, AC_LEFTMOST_LONGEST, LeftmostLongest);
testcombo!(search_leftmost_first, AC_LEFTMOST_FIRST, LeftmostFirst);
testcombo!(
    search_standard_nonoverlapping, AC_STANDARD_NON_OVERLAPPING, Standard
);

// Write out the overlapping combo by hand since there is only one of them.
testconfig!(
    overlapping,
    search_standard_overlapping_nfa_default,
    AC_STANDARD_OVERLAPPING,
    Standard,
    |_| ()
);
testconfig!(
    overlapping,
    search_standard_overlapping_nfa_all_sparse,
    AC_STANDARD_OVERLAPPING,
    Standard,
    |b: &mut AhoCorasickBuilder| { b.dense_depth(0); }
);
testconfig!(
    overlapping,
    search_standard_overlapping_nfa_all_dense,
    AC_STANDARD_OVERLAPPING,
    Standard,
    |b: &mut AhoCorasickBuilder| { b.dense_depth(usize::MAX); }
);
testconfig!(
    overlapping,
    search_standard_overlapping_dfa_default,
    AC_STANDARD_OVERLAPPING,
    Standard,
    |b: &mut AhoCorasickBuilder| { b.dfa(true); }
);
testconfig!(
    overlapping,
    search_standard_overlapping_dfa_all_sparse,
    AC_STANDARD_OVERLAPPING,
    Standard,
    |b: &mut AhoCorasickBuilder| { b.dfa(true).dense_depth(0); }
);
testconfig!(
    overlapping,
    search_standard_overlapping_dfa_all_dense,
    AC_STANDARD_OVERLAPPING,
    Standard,
    |b: &mut AhoCorasickBuilder| { b.dfa(true).dense_depth(usize::MAX); }
);
testconfig!(
    overlapping,
    search_standard_overlapping_dfa_no_byte_class,
    AC_STANDARD_OVERLAPPING,
    Standard,
    |b: &mut AhoCorasickBuilder| { b.dfa(true).byte_classes(false); }
);
testconfig!(
    overlapping,
    search_standard_overlapping_dfa_no_premultiply,
    AC_STANDARD_OVERLAPPING,
    Standard,
    |b: &mut AhoCorasickBuilder| { b.dfa(true).premultiply(false); }
);
testconfig!(
    overlapping,
    search_standard_overlapping_dfa_no_byte_class_no_premultiply,
    AC_STANDARD_OVERLAPPING,
    Standard,
    |b: &mut AhoCorasickBuilder| {
        b.dfa(true).byte_classes(false).premultiply(false);
    }
);

// Also write out tests manually for streams, since we only test the standard
// match semantics. We also don't bother testing different automaton
// configurations, since those are well covered by tests above.
testconfig!(
    stream,
    search_standard_stream_nfa_default,
    AC_STANDARD_NON_OVERLAPPING,
    Standard,
    |_| ()
);
testconfig!(
    stream,
    search_standard_stream_dfa_default,
    AC_STANDARD_NON_OVERLAPPING,
    Standard,
    |b: &mut AhoCorasickBuilder| { b.dfa(true); }
);

#[test]
fn search_tests_have_unique_names() {
    let assert = |constname, tests: &[SearchTest]| {
        let mut seen = HashMap::new(); // map from test name to position
        for (i, test) in tests.iter().enumerate() {
            if !seen.contains_key(test.name) {
                seen.insert(test.name, i);
            } else {
                let last = seen[test.name];
                panic!(
                    "{} tests have duplicate names at positions {} and {}",
                    constname, last, i
                );
            }
        }
    };
    assert("BASICS", BASICS);
    assert("STANDARD", STANDARD);
    assert("LEFTMOST", LEFTMOST);
    assert("LEFTMOST_FIRST", LEFTMOST_FIRST);
    assert("LEFTMOST_LONGEST", LEFTMOST_LONGEST);
    assert("NON_OVERLAPPING", NON_OVERLAPPING);
    assert("OVERLAPPING", OVERLAPPING);
    assert("REGRESSION", REGRESSION);
}

#[test]
#[should_panic]
fn stream_not_allowed_leftmost_first() {
    let fsm = AhoCorasickBuilder::new()
        .match_kind(MatchKind::LeftmostFirst)
        .build(None::<String>);
    assert_eq!(fsm.stream_find_iter(&b""[..]).count(), 0);
}

#[test]
#[should_panic]
fn stream_not_allowed_leftmost_longest() {
    let fsm = AhoCorasickBuilder::new()
        .match_kind(MatchKind::LeftmostLongest)
        .build(None::<String>);
    assert_eq!(fsm.stream_find_iter(&b""[..]).count(), 0);
}

#[test]
#[should_panic]
fn overlapping_not_allowed_leftmost_first() {
    let fsm = AhoCorasickBuilder::new()
        .match_kind(MatchKind::LeftmostFirst)
        .build(None::<String>);
    assert_eq!(fsm.find_overlapping_iter("").count(), 0);
}

#[test]
#[should_panic]
fn overlapping_not_allowed_leftmost_longest() {
    let fsm = AhoCorasickBuilder::new()
        .match_kind(MatchKind::LeftmostLongest)
        .build(None::<String>);
    assert_eq!(fsm.find_overlapping_iter("").count(), 0);
}

#[test]
fn state_id_too_small() {
    let mut patterns = vec![];
    for c1 in (b'a'..b'z').map(|b| b as char) {
        for c2 in (b'a'..b'z').map(|b| b as char) {
            for c3 in (b'a'..b'z').map(|b| b as char) {
                patterns.push(format!("{}{}{}", c1, c2, c3));
            }
        }
    }
    let result = AhoCorasickBuilder::new()
        .build_with_size::<u8, _, _>(&patterns);
    assert!(result.is_err());
}

fn run_search_tests<F: FnMut(&SearchTest) -> Vec<Match>>(
    which: TestCollection,
    mut f: F,
) {
    let get_match_triples =
        |matches: Vec<Match>| -> Vec<(usize, usize, usize)> {
            matches.into_iter()
                .map(|m| (m.pattern(), m.start(), m.end()))
                .collect()
        };
    for &tests in which {
        for test in tests {
            assert_eq!(
                test.matches,
                get_match_triples(f(&test)).as_slice(),
                "test: {}, patterns: {:?}, haystack: {:?}",
                test.name,
                test.patterns,
                test.haystack
            );
        }
    }
}
