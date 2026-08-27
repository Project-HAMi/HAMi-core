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

int main() {
    printf("Running getextrapid tests...\n");
    test_getextrapid_underflow();
    test_getextrapid_boundary();
    test_getextrapid_happy_path();
    test_getextrapid_empty();

    printf("Running mergepid tests...\n");
    test_mergepid_no_duplicates();
    test_mergepid_with_duplicates();

    printf("pid discovery tests passed\n");
    return 0;
}
