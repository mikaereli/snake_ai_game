import pygame
import random
import numpy as np

# Параметры игры
WIDTH = 600
HEIGHT = 600
GRID_SIZE = 20
GRID_WIDTH = WIDTH // GRID_SIZE
GRID_HEIGHT = HEIGHT // GRID_SIZE

# Цвета
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Направления движения
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

class Snake:
    def __init__(self):
        self.reset()

    def reset(self):
        self.length = 1
        self.positions = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])

    def get_head_position(self):
        return self.positions[0]

    def move(self, action):
        x, y = self.get_head_position()
        self.direction = action
        new_position = (x + self.direction[0], y + self.direction[1])
        self.positions.insert(0, new_position)
        if len(self.positions) > self.length:
            self.positions.pop()

    def check_collision(self):
        if self.check_boundary_collision():
            return True
        x, y = self.get_head_position()
        if (x, y) in self.positions[1:]:
            return True
        return False
    
    def check_boundary_collision(self):
        x, y = self.get_head_position()
        if x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT:
            return True
        return False

class Food:
    def __init__(self):
        self.position = (0, 0)
        self.randomize_position()

    def randomize_position(self):
        self.position = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))

class SnakeGame:
    def __init__(self):
        self.snake = Snake()
        self.food = Food()
        self.score = 0
        self.prev_distance_to_food = np.inf

    def update(self, action):
        head_x, head_y = self.snake.get_head_position()
        food_x, food_y = self.food.position
        self.prev_distance_to_food = np.sqrt((head_x - food_x)**2 + (head_y - food_y)**2)
        self.snake.move(action)
        if self.snake.get_head_position() == self.food.position:
            self.snake.length += 1
            self.score += 1
            self.food.randomize_position()
        if self.snake.check_collision():
            self.game_over = True

    def get_state(self):
        head_x, head_y = self.snake.get_head_position()
        food_x, food_y = self.food.position
        state = [
            head_x, head_y,
            food_x, food_y,
            self.snake.direction[0], self.snake.direction[1],
            self.snake.length
        ]
        return np.array(state, dtype=int)

    def get_reward(self):
        if self.game_over:
            if self.snake.check_boundary_collision():
                return -20  # Большой штраф за столкновение с границей
            else:
                return -10  # Штраф за столкновение с самим собой
        elif self.snake.get_head_position() == self.food.position:
            return 10
        else:
            head_x, head_y = self.snake.get_head_position()
            food_x, food_y = self.food.position
            distance_to_food = np.sqrt((head_x - food_x)**2 + (head_y - food_y)**2)
            if distance_to_food < self.prev_distance_to_food:
                return 1  # Вознаграждение за приближение к еде
            else:
                return -1  # Штраф за удаление от еды
    

class QLearningAgent:
    def __init__(self, actions):
        self.actions = actions
        self.q_table = np.zeros((GRID_WIDTH, GRID_HEIGHT, 4))
        self.epsilon = 1.0
        self.alpha = 0.5
        self.gamma = 0.9

    def get_action(self, state, epsilon):
        if np.random.uniform(0, 1) < epsilon:
            action = random.choice(self.actions)
        else:
            x, y = state[0], state[1]
            q_values = self.q_table[x, y]
            action_index = np.argmax(q_values)
            action = self.actions[action_index]
        return action
    
    def update_q_table(self, state, action, reward, next_state, game_over):
        x, y = state[0], state[1]
        action_index = self.actions.index(action)
        
        if game_over:
            self.q_table[x, y, action_index] = (1 - self.alpha) * self.q_table[x, y, action_index] + self.alpha * reward
        else:
            next_x, next_y = next_state[0], next_state[1]
            self.q_table[x, y, action_index] = (1 - self.alpha) * self.q_table[x, y, action_index] + \
                                                self.alpha * (reward + self.gamma * np.max(self.q_table[next_x, next_y]))

def train_agent(agent, game, num_episodes, epsilon_decay=0.995, min_epsilon=0.01):
    pygame.init()
    scores = []
    avg_scores = []
    epsilon = 1.0
    
    for episode in range(num_episodes):
        game.snake.reset()
        game.food.randomize_position()
        game.game_over = False
        state = game.get_state()
        score = 0
        
        while not game.game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return scores, avg_scores
            action = agent.get_action(state, epsilon)
            game.update(action)
            next_state = game.get_state()
            reward = game.get_reward()
            agent.update_q_table(state, action, reward, next_state, game.game_over)
            state = next_state
            score += reward

            screen = pygame.display.set_mode((WIDTH, HEIGHT))
            screen.fill(WHITE)
            for position in game.snake.positions:
                pygame.draw.rect(screen, GREEN, (position[0] * GRID_SIZE, position[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
            pygame.draw.rect(screen, RED, (game.food.position[0] * GRID_SIZE, game.food.position[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
            pygame.display.update()
            pygame.time.delay(50)
        
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)
        
        epsilon = max(epsilon * epsilon_decay, min_epsilon)
            
    return scores, avg_scores

def play_game(agent, game):
    game.snake.reset()
    game.food.randomize_position()
    game.game_over = False
    clock = pygame.time.Clock()
    while not game.game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        state = game.get_state()
        action = agent.get_action(state, 0.0)
        game.update(action)
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        screen.fill(WHITE)
        for position in game.snake.positions:
            pygame.draw.rect(screen, GREEN, (position[0] * GRID_SIZE, position[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        pygame.draw.rect(screen, RED, (game.food.position[0] * GRID_SIZE, game.food.position[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        pygame.display.update()
        clock.tick(10)

def main():
    pygame.init()
    game = SnakeGame()
    actions = [UP, DOWN, LEFT, RIGHT]
    agent = QLearningAgent(actions)
    num_episodes = 5000
    scores, avg_scores = train_agent(agent, game, num_episodes)
    play_game(agent, game)

if __name__ == "__main__":
    main()
