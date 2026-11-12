/**
 * CodeStorm 2026 - Algorithmic Question Dataset (Phase 3 Mock Questions)
 * Isolated question module with problem statements, constraints, testcases, and multi-language starter templates.
 */

export const starterCodeTemplates = {
  cpp: `#include <bits/stdc++.h>
using namespace std;

// Implement your solution below
class Solution {
public:
    void solve() {
        // Your code here
    }
};

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    Solution solver;
    solver.solve();
    
    return 0;
}
`,
  java: `import java.io.*;
import java.util.*;

public class Main {
    // Implement your solution below
    static class Solution {
        public void solve() {
            // Your code here
        }
    }

    public static void main(String[] args) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        Solution solver = new Solution();
        solver.solve();
    }
}
`,
  python: `import sys

# Implement your solution below
def solve():
    # Your code here
    pass

if __name__ == "__main__":
    solve()
`
};

export const questions = [
  {
    id: 1,
    title: "Two Sum",
    difficulty: "Easy",
    statement: `Given an array of integers <code>nums</code> and an integer <code>target</code>, return the indices of the two numbers such that they add up to <code>target</code>.

You may assume that each input would have <strong>exactly one solution</strong>, and you may not use the same element twice. You can return the answer in any order.`,
    inputFormat: `The first line contains an integer <code>n</code> denoting the size of the array.\nThe second line contains <code>n</code> space-separated integers representing the array elements.\nThe third line contains an integer <code>target</code>.`,
    outputFormat: `Print the two 0-based indices separated by a space.`,
    constraints: `2 &le; nums.length &le; 10<sup>5</sup>\n-10<sup>9</sup> &le; nums[i] &le; 10<sup>9</sup>\n-10<sup>9</sup> &le; target &le; 10<sup>9</sup>\nOnly one valid answer exists.`,
    sampleCases: [
      {
        input: `4\n2 7 11 15\n9`,
        output: `0 1`,
        explanation: `Because nums[0] + nums[1] == 9, we return 0 1.`
      },
      {
        input: `3\n3 2 4\n6`,
        output: `1 2`,
        explanation: `Because nums[1] + nums[2] == 6, we return 1 2.`
      }
    ],
    starterCode: {
      cpp: `#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    pair<int, int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int> seen;
        for (int i = 0; i < nums.size(); ++i) {
            int complement = target - nums[i];
            if (seen.count(complement)) {
                return {seen[complement], i};
            }
            seen[nums[i]] = i;
        }
        return {-1, -1};
    }
};

int main() {
    int n, target;
    if (!(cin >> n)) return 0;
    vector<int> nums(n);
    for (int i = 0; i < n; ++i) cin >> nums[i];
    cin >> target;

    Solution sol;
    auto [i1, i2] = sol.twoSum(nums, target);
    cout << i1 << " " << i2 << "\\n";
    return 0;
}
`,
      java: `import java.util.*;

public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        if (!sc.hasNextInt()) return;
        int n = sc.nextInt();
        int[] nums = new int[n];
        for (int i = 0; i < n; i++) nums[i] = sc.nextInt();
        int target = sc.nextInt();

        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < n; i++) {
            int comp = target - nums[i];
            if (map.containsKey(comp)) {
                System.out.println(map.get(comp) + " " + i);
                return;
            }
            map.put(nums[i], i);
        }
    }
}
`,
      python: `import sys

def solve():
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    nums = [int(x) for x in input_data[1:n+1]]
    target = int(input_data[n+1])

    seen = {}
    for i, num in enumerate(nums):
        comp = target - num
        if comp in seen:
            print(f"{seen[comp]} {i}")
            return
        seen[num] = i

if __name__ == "__main__":
    solve()
`
    }
  },
  {
    id: 2,
    title: "Valid Palindrome II",
    difficulty: "Easy",
    statement: `Given a string <code>s</code>, return <code>true</code> if the <code>s</code> can be palindrome after deleting <strong>at most one</strong> character from it.`,
    inputFormat: `A single line containing the string <code>s</code> consisting of lowercase English letters.`,
    outputFormat: `Print <code>true</code> or <code>false</code>.`,
    constraints: `1 &le; s.length &le; 10<sup>5</sup>\n<code>s</code> consists of lowercase English letters only.`,
    sampleCases: [
      {
        input: `aba`,
        output: `true`,
        explanation: `The string is already a palindrome.`
      },
      {
        input: `abca`,
        output: `true`,
        explanation: `You could delete the character 'c' to get 'aba'.`
      }
    ],
    starterCode: {
      cpp: `#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    bool isPalindromeRange(const string& s, int l, int r) {
        while (l < r) {
            if (s[l++] != s[r--]) return false;
        }
        return true;
    }

    bool validPalindrome(string s) {
        int l = 0, r = s.size() - 1;
        while (l < r) {
            if (s[l] != s[r]) {
                return isPalindromeRange(s, l + 1, r) || isPalindromeRange(s, l, r - 1);
            }
            l++;
            r--;
        }
        return true;
    }
};

int main() {
    string s;
    if (cin >> s) {
        Solution sol;
        cout << (sol.validPalindrome(s) ? "true" : "false") << "\\n";
    }
    return 0;
}
`,
      java: `import java.util.*;

public class Main {
    public static boolean check(String s, int l, int r) {
        while (l < r) {
            if (s.charAt(l++) != s.charAt(r--)) return false;
        }
        return true;
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        if (!sc.hasNext()) return;
        String s = sc.next();
        int l = 0, r = s.length() - 1;
        boolean ans = true;
        while (l < r) {
            if (s.charAt(l) != s.charAt(r)) {
                ans = check(s, l + 1, r) || check(s, l, r - 1);
                break;
            }
            l++;
            r--;
        }
        System.out.println(ans ? "true" : "false");
    }
}
`,
      python: `import sys

def valid_palindrome(s: str) -> bool:
    def check(l, r):
        while l < r:
            if s[l] != s[r]:
                return False
            l += 1
            r -= 1
        return True

    l, r = 0, len(s) - 1
    while l < r:
        if s[l] != s[r]:
            return check(l + 1, r) or check(l, r - 1)
        l += 1
        r -= 1
    return True

if __name__ == "__main__":
    line = sys.stdin.read().strip()
    if line:
        print("true" if valid_palindrome(line) else "false")
`
    }
  },
  {
    id: 3,
    title: "Longest Substring Without Repeating Characters",
    difficulty: "Medium",
    statement: `Given a string <code>s</code>, find the length of the <strong>longest substring</strong> without repeating characters.`,
    inputFormat: `A single line containing the string <code>s</code>.`,
    outputFormat: `Print a single integer representing the length of the longest substring with all unique characters.`,
    constraints: `0 &le; s.length &le; 5 &times; 10<sup>4</sup>\n<code>s</code> consists of English letters, digits, symbols and spaces.`,
    sampleCases: [
      {
        input: `abcabcbb`,
        output: `3`,
        explanation: `The answer is "abc", with the length of 3.`
      },
      {
        input: `bbbbb`,
        output: `1`,
        explanation: `The answer is "b", with the length of 1.`
      }
    ],
    starterCode: {
      cpp: `#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        vector<int> last(256, -1);
        int maxLen = 0, start = 0;
        for (int i = 0; i < s.length(); ++i) {
            if (last[(unsigned char)s[i]] >= start) {
                start = last[(unsigned char)s[i]] + 1;
            }
            last[(unsigned char)s[i]] = i;
            maxLen = max(maxLen, i - start + 1);
        }
        return maxLen;
    }
};

int main() {
    string s;
    if (getline(cin, s)) {
        Solution sol;
        cout << sol.lengthOfLongestSubstring(s) << "\\n";
    }
    return 0;
}
`,
      java: `import java.util.*;

public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        String s = sc.hasNextLine() ? sc.nextLine() : "";
        int[] last = new int[256];
        Arrays.fill(last, -1);
        int maxLen = 0, start = 0;
        for (int i = 0; i < s.length(); i++) {
            int c = s.charAt(i);
            if (last[c] >= start) {
                start = last[c] + 1;
            }
            last[c] = i;
            maxLen = Math.max(maxLen, i - start + 1);
        }
        System.out.println(maxLen);
    }
}
`,
      python: `import sys

def length_of_longest_substring(s: str) -> int:
    last = {}
    start = 0
    max_len = 0
    for i, char in enumerate(s):
        if char in last and last[char] >= start:
            start = last[char] + 1
        last[char] = i
        max_len = max(max_len, i - start + 1)
    return max_len

if __name__ == "__main__":
    line = sys.stdin.read().rstrip('\\r\\n')
    print(length_of_longest_substring(line))
`
    }
  },
  {
    id: 4,
    title: "Binary Tree Level Order Traversal",
    difficulty: "Medium",
    statement: `Given the root of a binary tree represented in level-order serialization (using <code>null</code> for missing nodes), return the <em>level order traversal</em> of its nodes' values (i.e., from left to right, level by level).`,
    inputFormat: `A single line containing space-separated node values in level-order, with <code>null</code> representing empty nodes.`,
    outputFormat: `Print each level on a new line with values separated by spaces.`,
    constraints: `The number of nodes in the tree is in the range [0, 2000].\n-1000 &le; Node.val &le; 1000.`,
    sampleCases: [
      {
        input: `3 9 20 null null 15 7`,
        output: `3\n9 20\n15 7`,
        explanation: `Level 1: [3], Level 2: [9, 20], Level 3: [15, 7].`
      }
    ],
    starterCode: {
      cpp: `#include <bits/stdc++.h>
using namespace std;

struct TreeNode {
    int val;
    TreeNode *left, *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

int main() {
    // Implement BFS level-order traversal
    return 0;
}
`,
      java: `import java.util.*;

public class Main {
    static class TreeNode {
        int val;
        TreeNode left, right;
        TreeNode(int x) { val = x; }
    }

    public static void main(String[] args) {
        // Implement level-order queue BFS
    }
}
`,
      python: `import sys
from collections import deque

def solve():
    # Implement BFS level order traversal
    pass

if __name__ == "__main__":
    solve()
`
    }
  },
  {
    id: 5,
    title: "Merge k Sorted Lists",
    difficulty: "Hard",
    statement: `You are given an array of <code>k</code> linked-lists <code>lists</code>, each linked-list is sorted in ascending order.

<em>Merge all the linked-lists into one sorted linked-list and return it.</em>`,
    inputFormat: `The first line contains an integer <code>k</code> denoting the number of sorted lists.\nThe next <code>k</code> lines each contain an integer <code>m</code> (the size of that list) followed by <code>m</code> space-separated sorted integers.`,
    outputFormat: `Print the merged sorted list values separated by spaces.`,
    constraints: `k == lists.length\n0 &le; k &le; 10<sup>4</sup>\n0 &le; lists[i].length &le; 500\n-10<sup>4</sup> &le; lists[i][j] &le; 10<sup>4</sup>\nThe sum of lists[i].length will not exceed 10<sup>4</sup>.`,
    sampleCases: [
      {
        input: `3\n3 1 4 5\n3 1 3 4\n2 2 6`,
        output: `1 1 2 3 4 4 5 6`,
        explanation: `The lists are [1,4,5], [1,3,4], [2,6]. Merging them in sorted order yields [1,1,2,3,4,4,5,6].`
      }
    ],
    starterCode: {
      cpp: `#include <bits/stdc++.h>
using namespace std;

int main() {
    int k;
    if (!(cin >> k)) return 0;
    priority_queue<int, vector<int>, greater<int>> pq;
    for (int i = 0; i < k; ++i) {
        int m; cin >> m;
        for (int j = 0; j < m; ++j) {
            int val; cin >> val;
            pq.push(val);
        }
    }
    bool first = true;
    while (!pq.empty()) {
        if (!first) cout << " ";
        cout << pq.top();
        pq.pop();
        first = false;
    }
    cout << "\\n";
    return 0;
}
`,
      java: `import java.util.*;

public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        if (!sc.hasNextInt()) return;
        int k = sc.nextInt();
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        for (int i = 0; i < k; i++) {
            int m = sc.nextInt();
            for (int j = 0; j < m; j++) {
                pq.add(sc.nextInt());
            }
        }
        StringBuilder sb = new StringBuilder();
        while (!pq.isEmpty()) {
            sb.append(pq.poll()).append(pq.isEmpty() ? "" : " ");
        }
        System.out.println(sb.toString());
    }
}
`,
      python: `import sys
import heapq

def solve():
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    k = int(input_data[0])
    idx = 1
    heap = []
    for _ in range(k):
        m = int(input_data[idx])
        idx += 1
        for _ in range(m):
            heapq.heappush(heap, int(input_data[idx]))
            idx += 1
    
    result = []
    while heap:
        result.append(str(heapq.heappop(heap)))
    print(" ".join(result))

if __name__ == "__main__":
    solve()
`
    }
  }
];
