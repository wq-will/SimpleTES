# EVOLVE-BLOCK-START
CPP_CODE = '''
#include <bits/stdc++.h>
using namespace std;
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int N, L, T;
    long long K;
    cin >> N >> L >> T >> K;
    for (int i = 0; i < N; ++i) { int a; cin >> a; }
    for (int i = 0; i < L; ++i)
        for (int j = 0; j < N; ++j) { long long c; cin >> c; }
    for (int t = 0; t < T; ++t) cout << "-1\\n";
    return 0;
}
'''
# EVOLVE-BLOCK-END