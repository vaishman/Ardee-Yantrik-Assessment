import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

class Node:
    def __init__(self, x, y):
        self.x = x  # Integer x-coordinate
        self.y = y  # Integer y-coordinate
        self.parent = None
        self.cost = 0.0

class RRTStar:
    def __init__(self, start, goal, grid, grid_size, obstacle_points, step_size=1, max_iter=1000, goal_radius=0, search_radius=3):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.grid = grid
        self.grid_size = grid_size
        self.obstacle_points = obstacle_points  # Store for visualization
        self.step_size = step_size
        self.max_iter = max_iter
        self.goal_radius = goal_radius
        self.search_radius = search_radius
        self.nodes = [self.start]

    def plan(self):
        # Initialize plot for real-time visualization
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 8))
        # Plot obstacle points as individual markers
        obs_x, obs_y = zip(*self.obstacle_points)
        ax.plot(obs_x, obs_y, 'kx', markersize=10, label='Obstacles')  # Black 'x' markers
        ax.plot(self.start.x, self.start.y, 'bo', markersize=10, label='Start')
        ax.plot(self.goal.x, self.goal.y, 'ro', markersize=10, label='Goal')
        ax.legend()
        ax.grid(True)
        ax.set_xlim(-0.5, self.grid_size[0] - 0.5)  # Match grid dimensions
        ax.set_ylim(-0.5, self.grid_size[1] - 0.5)
        ax.set_xticks(np.arange(0, self.grid_size[0], 1))
        ax.set_yticks(np.arange(0, self.grid_size[1], 1))
        ax.set_facecolor('#f0f0f0')  # Light gray background
        plt.title("RRT* Iteration: 0")
        
        for i in range(self.max_iter):
            # Sample a random integer point
            rand_point = self.sample()
            # Find nearest node
            nearest_node = self.nearest(rand_point)
            # Steer towards random point
            new_node = self.steer(nearest_node, rand_point)
            if new_node and self.is_collision_free(nearest_node, new_node):
                # Find nearby nodes for rewiring
                near_nodes = self.near_nodes(new_node)
                # Choose best parent
                min_cost = nearest_node.cost + self.distance(nearest_node, new_node)
                best_parent = nearest_node
                for near_node in near_nodes:
                    cost = near_node.cost + self.distance(near_node, new_node)
                    if self.is_collision_free(near_node, new_node) and cost < min_cost:
                        min_cost = cost
                        best_parent = near_node
                new_node.parent = best_parent
                new_node.cost = min_cost
                self.nodes.append(new_node)
                # Plot the new node and its connection
                ax.plot(new_node.x, new_node.y, 'g.', markersize=5)  # Plot node as green dot
                ax.plot([new_node.x, new_node.parent.x], [new_node.y, new_node.parent.y], 'g-', alpha=0.5)
                # Visualize every 10 iterations
                if i % 10 == 0 or self.distance(new_node, self.goal) <= self.goal_radius:
                    plt.draw()
                    plt.pause(0.01)
                # Check if goal is reached
                if self.distance(new_node, self.goal) <= self.goal_radius:
                    if self.is_collision_free(new_node, self.goal):
                        self.goal.parent = new_node
                        self.goal.cost = new_node.cost + self.distance(new_node, self.goal)
                        self.nodes.append(self.goal)
                        ax.plot(self.goal.x, self.goal.y, 'g.', markersize=5)  # Plot goal node
                        ax.plot([self.goal.x, new_node.x], [self.goal.y, new_node.y], 'g-', alpha=0.5)
                        path = self.get_path()
                        path_np = np.array(path)
                        ax.plot(path_np[:, 0], path_np[:, 1], 'r-', linewidth=2)
                        plt.title(f"RRT* Iteration: {i+1} (Path Found)")
                        print(f"Path found after {i+1} iterations")
                        print("Final Path:", path)
                        plt.draw()
                        plt.pause(0.1)
                        plt.savefig('rrt_star_final.png')
                        plt.ioff()
                        plt.show()
                        return path
                # Update plot title every 10 iterations
                if i % 10 == 0:
                    plt.title(f"RRT* Iteration: {i+1}")
                    print(f"Iteration: {i+1}")
        print("No path found within max iterations")
        plt.ioff()
        plt.savefig('rrt_star_final.png')
        plt.show()
        return None

    def sample(self):
        if np.random.random() < 0.1:
            return [self.goal.x, self.goal.y]
        # Generate random integer coordinates
        return [np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1])]

    def nearest(self, point):
        min_dist = float('inf')
        nearest = None
        for node in self.nodes:
            dist = self.distance(node, point)
            if dist < min_dist:
                min_dist = dist
                nearest = node
        return nearest

    def steer(self, from_node, to_point):
        dist = self.distance(from_node, to_point)
        if dist <= self.step_size:
            return Node(to_point[0], to_point[1])
        # Move step_size in the direction of to_point, rounding to integer
        theta = np.arctan2(to_point[1] - from_node.y, to_point[0] - from_node.x)
        x = int(round(from_node.x + self.step_size * np.cos(theta)))
        y = int(round(from_node.y + self.step_size * np.sin(theta)))
        # Clamp to grid bounds
        x = max(0, min(x, self.grid_size[0] - 1))
        y = max(0, min(y, self.grid_size[1] - 1))
        return Node(x, y)

    def distance(self, node1, point_or_node):
        if isinstance(point_or_node, Node):
            return sqrt((node1.x - point_or_node.x)**2 + (node1.y - point_or_node.y)**2)
        return sqrt((node1.x - point_or_node[0])**2 + (node1.y - point_or_node[1])**2)

    def is_collision_free(self, node1, node2):
        # Check if nodes are in bounds and not in obstacles
        for node in [node1, node2]:
            x, y = node.x, node.y
            if not (0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]):
                return False
            if self.grid[x, y] == 1:
                return False
        # Bresenham's line algorithm to check all grid cells between nodes
        x1, y1 = node1.x, node1.y
        x2, y2 = node2.x, node2.y
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        x, y = x1, y1
        err = dx - dy
        while True:
            # Check current grid cell
            if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
                if self.grid[x, y] == 1:
                    return False
            else:
                return False
            # Check if reached the end point
            if x == x2 and y == y2:
                break
            # Move to next grid cell
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return True

    def near_nodes(self, node):
        return [n for n in self.nodes if self.distance(n, node) <= self.search_radius]

    def get_path(self):
        path = []
        current = self.goal
        while current:
            path.append([current.x, current.y])
            current = current.parent
        return path[::-1]

def main():
    # Create a 20x20 grid with point obstacles
    grid_size = (20, 20)
    grid = np.zeros(grid_size, dtype=int)
    # Define point obstacles (integer coordinates)
    obstacle_points = [
        [6, 6], [6, 7], [6, 8], [6, 9],  # Cluster of points
        [10, 10], [11, 10], [12, 10],     # Horizontal points
        [15, 15], [15, 16], [16, 15]      # Diagonal points
    ]
    for x, y in obstacle_points:
        grid[x, y] = 1
    start = [1, 1]  # Integer coordinates
    goal = [18, 18]  # Integer coordinates
    # Verify start and goal are not in obstacles
    if grid[start[0], start[1]] == 1:
        print("Error: Start point is in an obstacle")
        return
    if grid[goal[0], goal[1]] == 1:
        print("Error: Goal point is in an obstacle")
        return
    rrt_star = RRTStar(start, goal, grid, grid_size, obstacle_points, step_size=1, max_iter=1000, goal_radius=0, search_radius=3)
    path = rrt_star.plan()
    if path:
        print("Final Path Coordinates:", path)
        # Verify path points
        for x, y in path:
            if 0 <= x < grid_size[0] and 0 <= y < grid_size[1]:
                if grid[x, y] == 1:
                    print(f"Warning: Path point ({x}, {y}) lies in obstacle")
            else:
                print(f"Warning: Path point ({x}, {y}) is out of bounds")
    else:
        print("No path found")

if __name__ == "__main__":
    main()