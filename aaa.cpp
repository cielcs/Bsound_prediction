#include <iostream>
#include <vector>
#include <queue>
#include <cmath>

using namespace std;


int count_points(int N, int D, const vector<pair<int, int>>& points) {
    vector<vector<int>> distances(N, vector<int>(N, 0));
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            distances[i][j] = abs(points[i].first - points[j].first) + abs(points[i].second - points[j].second);
            distances[j][i] = distances[i][j];
        }
    }
    
    vector<bool> visited(N, false);
    queue<int> q;
    int count = 0;
    
    for (int i = 0; i < N; ++i) {
        if (!visited[i]) {
            q.push(i);
            visited[i] = true;
            
            while (!q.empty()) {
                int curr = q.front();
                q.pop();
                
                for (int j = 0; j < N; ++j) {
                    if (!visited[j] && distances[curr][j] <= D) {
                        q.push(j);
                        visited[j] = true;
                    }
                }
            }
            
            count++;
        }
    }
    
    return count;
}


int main() {
    int N, D;
    cin >> N >> D;
    
    vector<pair<int, int>> points(N);
    for (int i = 0; i < N; ++i) {
        cin >> points[i].first >> points[i].second;
    }
    
    int result = count_points(N, D, points);
    cout << result << endl;

    
    return 0;
}
