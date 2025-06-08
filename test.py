
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0

class RRTStar:
    def __init__(self, start, goal, grid, grid_size, step_size=1.0, max_iter=500, goal_radius=1.0, search_radius=2.0):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.grid = grid
        self.grid_size = grid_size
        self.step_size = step_size
        self.max_iter = max_iter
        self.goal_radius = goal_radius
        self.search_radius = search_radius
        self.nodes = [self.start]

    def plan(self):
        # Initialize plot for real-time visualization
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(self.grid, cmap='binary', origin='lower')
        ax.plot(self.start.x, self.start.y, 'bo', label='Start')
        ax.plot(self.goal.x, self.goal.y, 'ro', label='Goal')
        ax.legend()
        ax.grid(True)
        plt.title("RRT* Iteration: 0")
        
        for i in range(self.max_iter):
            # Sample a random point
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
                # Rewire the tree
                for near_node in near_nodes:
                    cost = new_node.cost + self.distance(new_node, near_node)
                    if self.is_collision_free(new_node, near_node) and cost < near_node.cost:
                        near_node.parent = new_node
                        near_node.cost = cost
                # Visualize the new edge
                ax.plot([new_node.x, new_node.parent.x], [new_node.y, new_node.parent.y], 'g-', alpha=0.5)
                # Check if goal is reached
                if self.distance(new_node, self.goal) < self.goal_radius:
                    if self.is_collision_free(new_node, self.goal):
                        self.goal.parent = new_node
                        self.goal.cost = new_node.cost + self.distance(new_node, self.goal)
                        self.nodes.append(self.goal)
                        path = self.get_path()
                        path_np = np.array(path)
                        ax.plot(path_np[:, 0], path_np[:, 1], 'r-', linewidth=2)
                        plt.title(f"RRT* Iteration: {i+1} (Path Found)")
                        print(f"Path found after {i+1} iterations")
                        plt.draw()
                        plt.pause(0.1)
                        plt.savefig('rrt_star_final.png')
                        plt.ioff()
                        plt.show()
                        return path
                # Update plot
                plt.title(f"RRT* Iteration: {i+1}")
                print(f"Iteration: {i+1}")
                plt.draw()
                plt.pause(0.01)  # Brief pause to allow plot update
        print("No path found within max iterations")
        plt.ioff()
        plt.savefig('rrt_star_final.png')
        plt.show()
        return None

    def sample(self):
        if np.random.random() < 0.01:  # Bias towards goal
            return [self.goal.x, self.goal.y]
        return [np.random.uniform(0, self.grid_size[0]), np.random.uniform(0, self.grid_size[1])]

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
        if dist < self.step_size:
            return Node(to_point[0], to_point[1])
        theta = np.arctan2(to_point[1] - from_node.y, to_point[0] - from_node.x)
        x = from_node.x + self.step_size * np.cos(theta)
        y = from_node.y + self.step_size * np.sin(theta)
        return Node(x, y)

    def distance(self, node1, point_or_node):
        if isinstance(point_or_node, Node):
            return sqrt((node1.x - point_or_node.x)**2 + (node1.y - point_or_node.y)**2)
        return sqrt((node1.x - point_or_node[0])**2 + (node1.y - point_or_node[1])**2)

    def is_collision_free(self, node1, node2):
        # Check if nodes are in bounds and not in obstacles
        for node in [node1, node2]:
            x, y = int(node.x), int(node.y)
            if not (0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]):
                return False
            if self.grid[x, y] == 1:
                return False
        # Check line segment for collisions using fine sampling
        num_steps = int(self.distance(node1, node2) / 0.01) + 1  # Finer resolution
        for i in range(num_steps + 1):
            t = i / num_steps
            x = node1.x + t * (node2.x - node1.x)
            y = node1.y + t * (node2.y - node1.y)
            x_int, y_int = int(x), int(y)
            if 0 <= x_int < self.grid_size[0] and 0 <= y_int < self.grid_size[1]:
                if self.grid[x_int, y_int] == 1:
                    return False
            else:
                return False
        return True

    def near_nodes(self, node):
        return [n for n in self.nodes if self.distance(n, node) < self.search_radius]

    def get_path(self):
        path = []
        current = self.goal
        while current:
            path.append([current.x, current.y])
            current = current.parent
        return path[::-1]

def main():
    # Create a 20x20 grid with obstacles
    grid_size = (20, 20)
    grid = np.zeros(grid_size)
    grid[5:7, 5:15] = 1  # Horizontal obstacle
    grid[10:12, 5:10] = 1  # Rectangular obstacle
    grid[15:17, 10:15] = 1  # Another obstacle
    start = [1, 1]
    goal = [18, 18]
    rrt_star = RRTStar(start, goal, grid, grid_size, step_size=1.0, max_iter=1000, goal_radius=1.0, search_radius=2.0)
    path = rrt_star.plan()
    if path:
        print("Final Path:", path)
    else:
        print("No path found")

if __name__ == "__main__":
    main()