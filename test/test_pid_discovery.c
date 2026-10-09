#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/include/utils.h"

// Simple assertion macro
#define ASSERT_EQ(actual, expected) \
    if ((actual) != (expected)) { \
        fprintf(stderr, "%s:%d: Assertion failed: %d != %d\n", __FILE__, __LINE__, (int)(actual), (int)(expected)); \
        exit(1); \
    }

void test_getextrapid_underflow() {
    nvmlProcessInfo_t1 pre[] = { {100, 0}, {101, 0}, {102, 0} };
    nvmlProcessInfo_t1 cur[] = { {999, 0} };
    int extra = getextrapid(3, 1, pre, cur);
    ASSERT_EQ(extra, 0); // underflow should be handled and return 0
}

void test_getextrapid_boundary() {
    nvmlProcessInfo_t1 pre[] = { {100, 0}, {101, 0} };
    nvmlProcessInfo_t1 cur[] = { {100, 0}, {101, 0} };
    int extra = getextrapid(2, 2, pre, cur);
    ASSERT_EQ(extra, 0); // same elements
}

void test_getextrapid_happy_path() {
    nvmlProcessInfo_t1 pre[] = { {100, 0} };
    nvmlProcessInfo_t1 cur[] = { {100, 0}, {102, 0} };
    int extra = getextrapid(1, 2, pre, cur);
    ASSERT_EQ(extra, 102); // 102 is the new one
}

void test_getextrapid_empty() {
    nvmlProcessInfo_t1 *pre = NULL;
    nvmlProcessInfo_t1 cur[] = { {103, 0} };
    int extra = getextrapid(0, 1, pre, cur);
    ASSERT_EQ(extra, 103);

    int extra2 = getextrapid(0, 0, pre, cur);
    ASSERT_EQ(extra2, 0);
}

void test_mergepid_no_duplicates() {
    nvmlProcessInfo_t1 sub[] = { {100, 0}, {101, 0} };
    nvmlProcessInfo_t1 merged[10];
    unsigned int prev = 2;
    unsigned int current = 0;

    mergepid(&prev, &current, sub, merged);
    
    ASSERT_EQ(current, 2);
    ASSERT_EQ(merged[0].pid, 100);
    ASSERT_EQ(merged[1].pid, 101);
}

void test_mergepid_with_duplicates() {
    nvmlProcessInfo_t1 sub[] = { {100, 0}, {102, 0} };
    nvmlProcessInfo_t1 merged[10] = { {100, 0}, {101, 0} };
    unsigned int prev = 2;
    unsigned int current = 2;

    mergepid(&prev, &current, sub, merged);
    
    ASSERT_EQ(current, 3); // 100 already exists, 102 is added
    ASSERT_EQ(merged[0].pid, 100);
    ASSERT_EQ(merged[1].pid, 101);
    ASSERT_EQ(merged[2].pid, 102);
}

void test_own_pid_alone() {
    nvmlProcessInfo_t1 before[] = { {100, 500} };
    nvmlProcessInfo_t1 during[] = { {100, 500}, {200, 248} };
    nvmlProcessInfo_t1 after[] = { {100, 500} };
    unsigned int out[4] = {0};
    ASSERT_EQ(own_pid_candidates(before, 1, during, 2, after, 1, NULL, 0, out, 4), 1);
    ASSERT_EQ(out[0], 200);
}

void test_own_pid_neighbour_starts() {
    nvmlProcessInfo_t1 before[] = { {100, 500} };
    nvmlProcessInfo_t1 during[] = { {100, 500}, {300, 2048}, {200, 248} };
    nvmlProcessInfo_t1 after[] = { {100, 500}, {300, 2048} };
    unsigned int out[4] = {0};
    ASSERT_EQ(getextrapid(1, 3, before, during), 300);
    ASSERT_EQ(own_pid_candidates(before, 1, during, 3, after, 2, NULL, 0, out, 4), 1);
    ASSERT_EQ(out[0], 200);
}

void test_own_pid_narrowed_by_earlier_probes() {
    nvmlProcessInfo_t1 before[] = { {100, 500} };
    nvmlProcessInfo_t1 during[] = { {100, 500}, {300, 248}, {200, 248} };
    nvmlProcessInfo_t1 after[] = { {100, 500} };
    unsigned int out[4] = {0};
    ASSERT_EQ(own_pid_candidates(before, 1, during, 3, after, 1, NULL, 0, out, 4), 2);
    nvmlProcessInfo_t1 during2[] = { {100, 500}, {200, 248}, {400, 248} };
    unsigned int known[2] = {300, 200};
    ASSERT_EQ(own_pid_candidates(before, 1, during2, 3, after, 1, known, 2, out, 4), 1);
    ASSERT_EQ(out[0], 200);
}

void test_own_pid_narrowing_keeps_every_repeat_match() {
    nvmlProcessInfo_t1 before[] = { {100, 500} };
    nvmlProcessInfo_t1 during[] = { {100, 500}, {200, 248}, {300, 248} };
    nvmlProcessInfo_t1 after[] = { {100, 500} };
    unsigned int known[2] = {300, 200};
    unsigned int out[4] = {0};
    ASSERT_EQ(own_pid_candidates(before, 1, during, 3, after, 1, known, 2, out, 4), 2);
    ASSERT_EQ(out[0], 200);
    ASSERT_EQ(out[1], 300);
    ASSERT_EQ(known[0], 300);  // `known` is only read
}

void test_own_pid_restarts_when_known_matches_nothing() {
    nvmlProcessInfo_t1 before[] = { {100, 500} };
    nvmlProcessInfo_t1 during[] = { {100, 500}, {200, 248} };
    nvmlProcessInfo_t1 after[] = { {100, 500} };
    unsigned int known[1] = {300};
    unsigned int out[4] = {0};
    ASSERT_EQ(own_pid_candidates(before, 1, during, 2, after, 1, known, 1, out, 4), 1);
    ASSERT_EQ(out[0], 200);
}

void test_own_pid_nothing_appeared() {
    nvmlProcessInfo_t1 before[] = { {100, 500}, {200, 248} };
    nvmlProcessInfo_t1 during[] = { {100, 500}, {200, 248} };
    nvmlProcessInfo_t1 after[] = { {100, 500}, {200, 248} };
    unsigned int out[4] = {0};
    ASSERT_EQ(own_pid_candidates(before, 2, during, 2, after, 2, NULL, 0, out, 4), 0);
    ASSERT_EQ(own_pid_candidates(before, 2, during, 0, after, 0, NULL, 0, out, 4), 0);
}

void test_own_pid_neighbour_exits_or_listed_twice() {
    nvmlProcessInfo_t1 before[] = { {100, 500}, {150, 600} };
    nvmlProcessInfo_t1 during[] = { {100, 500}, {200, 248}, {200, 248} };
    nvmlProcessInfo_t1 after[] = { {100, 500} };
    unsigned int out[4] = {0};
    ASSERT_EQ(own_pid_candidates(before, 2, during, 3, after, 1, NULL, 0, out, 4), 1);
    ASSERT_EQ(out[0], 200);
}

void test_retry_seed_differs_for_same_pid_same_second() {
    struct timespec a = {1000, 111}, b = {1000, 222};
    ASSERT_EQ(hostpid_retry_seed(&a, 7) != hostpid_retry_seed(&b, 7), 1);
    ASSERT_EQ(hostpid_retry_seed(&a, 7) != hostpid_retry_seed(&a, 8), 1);
}

int main() {
    printf("Running getextrapid tests...\n");
    test_getextrapid_underflow();
    test_getextrapid_boundary();
    test_getextrapid_happy_path();
    test_getextrapid_empty();

    printf("Running own_pid_candidates tests...\n");
    test_own_pid_alone();
    test_own_pid_neighbour_starts();
    test_own_pid_narrowed_by_earlier_probes();
    test_own_pid_narrowing_keeps_every_repeat_match();
    test_own_pid_restarts_when_known_matches_nothing();
    test_own_pid_nothing_appeared();
    test_own_pid_neighbour_exits_or_listed_twice();

    test_retry_seed_differs_for_same_pid_same_second();

    printf("Running mergepid tests...\n");
    test_mergepid_no_duplicates();
    test_mergepid_with_duplicates();

    printf("pid discovery tests passed\n");
    return 0;
}
