from __future__ import annotations
from dataclasses import dataclass
from math import sqrt, pi

@dataclass(slots=True)
class Point:
    x: float
    y: float
    
    def distance_to(self, target_point: Point) -> float:
        return sqrt((self.x - target_point.x) ** 2 + (self.y - target_point.y) ** 2)

@dataclass(slots=True)
class Vector:
    x: float
    y: float
    
    def magnitude(self) -> float:
        return sqrt(self.x * self.x + self.y * self.y)
    
    def normalize(self) -> Vector:
        magnitude = self.magnitude()
        if magnitude == 0:
            return Vector(0, 0)
        return Vector(self.x / magnitude, self.y / magnitude)
    
    def dot_product(self, target_vector: Vector) -> float:
        return self.x * target_vector.x + self.y * target_vector.y
    
    def is_parallel_to(self, target_vector: Vector) -> bool:
        if self.magnitude() == 0 or target_vector.magnitude() == 0:
            return True
        vector1_normalized = self.normalize()
        vector2_normalized = target_vector.normalize()
        dot_product_result = abs(vector1_normalized.dot_product(vector2_normalized))
        return abs(dot_product_result - 1) < 1e-10

@dataclass(slots=True)
class Line:
    start: Point
    end: Point
    
    def to_vector(self) -> Vector:
        return Vector(self.end.x - self.start.x, self.end.y - self.start.y)
    
    def is_parallel_to(self, target_line: Line) -> bool:
        return self.to_vector().is_parallel_to(target_line.to_vector())
    
    def is_perpendicular_to(self, target_line: Line) -> bool:
        dot_product_result = self.to_vector().dot_product(target_line.to_vector())
        return abs(dot_product_result) < 1e-10

@dataclass(slots=True)
class Circle:
    center: Point
    radius: float
    
    def get_area(self) -> float:
        return pi * self.radius ** 2
    
    def intersects_with(self, target_circle: Circle) -> bool:
        centers_distance = self.center.distance_to(target_circle.center)
        radius_sum = self.radius + target_circle.radius
        radius_difference = abs(self.radius - target_circle.radius)
        return radius_difference < centers_distance < radius_sum

@dataclass(slots=True)
class Polygon:
    points: list[Point]
    
    def get_perimeter(self) -> float:
        perimeter = 0.0
        points_count = len(self.points)
        
        for i in range(points_count):
            current_point = self.points[i]
            next_point = self.points[(i + 1) % points_count]
            perimeter += current_point.distance_to(next_point)
            
        return perimeter

class Coordinate:
    @staticmethod
    def create_point(x: float, y: float) -> Point:
        return Point(x, y)
    
    @staticmethod
    def create_line(start_x: float, start_y: float, end_x: float, end_y: float) -> Line:
        start_point = Point(start_x, start_y)
        end_point = Point(end_x, end_y)
        return Line(start_point, end_point)
    
    @staticmethod
    def create_circle(center_x: float, center_y: float, radius: float) -> Circle:
        center_point = Point(center_x, center_y)
        return Circle(center_point, radius)
    
    @staticmethod
    def create_polygon(points: list[tuple[float, float]]) -> Polygon:
        polygon_points = [Point(x, y) for x, y in points]
        return Polygon(polygon_points)

@dataclass(slots=True)
class Enemy:
    position: Point
    direction: Vector
    life_points: int = 10
    
    @property
    def is_dead(self) -> bool:
        return self.life_points <= 0
    
    def move_forward(self) -> None:
        if not self.is_dead:
            target_position_x = self.position.x + self.direction.x
            target_position_y = self.position.y + self.direction.y
            self.position = Point(target_position_x, target_position_y)
    
    def take_damage(self, damage: int) -> None:
        if not self.is_dead:
            self.life_points -= damage

class Tower(Circle):
    def __init__(self, center: Point, radius: float, power: int):
        super().__init__(center=center, radius=radius)
        self.power = power
    
    def is_in_range(self, enemy: Enemy) -> bool:
        if enemy.is_dead:
            return False
        return self.center.distance_to(enemy.position) <= self.radius
    
    def attack(self, enemy: Enemy) -> bool:
        if self.is_in_range(enemy):
            enemy.take_damage(self.power)
            return True
        return False

class BasicTower(Tower):
    def __init__(self, center: Point):
        super().__init__(center=center, radius=2, power=1)

class AdvancedTower(Tower):
    def __init__(self, center: Point):
        super().__init__(center=center, radius=4, power=2)

class WarMap:
    def __init__(self):
        self.coordinate = Coordinate()
        self.enemies: list[Enemy] = []
        self.basic_towers: list[BasicTower] = []
        self.advanced_towers: list[AdvancedTower] = []
    
    def add_enemy(self, position: tuple[float, float], direction: tuple[float, float]) -> Enemy:
        enemy_position = self.coordinate.create_point(position[0], position[1])
        enemy_direction = Vector(direction[0], direction[1])
        enemy = Enemy(enemy_position, enemy_direction)
        self.enemies.append(enemy)
        return enemy
    
    def add_basic_tower(self, x: float, y: float) -> BasicTower:
        tower_position = self.coordinate.create_point(x, y)
        tower = BasicTower(tower_position)
        self.basic_towers.append(tower)
        return tower
    
    def add_advanced_tower(self, x: float, y: float) -> AdvancedTower:
        tower_position = self.coordinate.create_point(x, y)
        tower = AdvancedTower(tower_position)
        self.advanced_towers.append(tower)
        return tower

class Game:
    def __init__(self, war_map: WarMap):
        self.war_map = war_map
        self.current_turn = 0
    
    def execute_turn(self) -> None:
        self.move_enemies()
        self.towers_attack()
        self.current_turn += 1
    
    def move_enemies(self) -> None:
        for enemy in self.war_map.enemies:
            enemy.move_forward()
    
    def towers_attack(self) -> None:
        for tower in self.war_map.basic_towers + self.war_map.advanced_towers:
            for enemy in self.war_map.enemies:
                tower.attack(enemy)
    
    def run(self, turns: int = 10) -> None:
        for _ in range(turns):
            self.execute_turn()
        self.print_result()
    
    def print_result(self) -> None:
        for index, enemy in enumerate(self.war_map.enemies, 1):
            status = '(Dead)' if enemy.is_dead == True else ''
            print(f'{status} E{index}: Position({enemy.position.x}, {enemy.position.y}), Life Points: {enemy.life_points}')


def execute_task1():
    coordinate = Coordinate()
    line_a = coordinate.create_line(-6, 1, 2, 4)
    line_b = coordinate.create_line(-6, -1, 2, 2)
    parallel_result = line_a.is_parallel_to(line_b)
    print(f'Are Line A and Line B parallel? {parallel_result}')
    
    line_c = coordinate.create_line(-4, -4, -1, 6)
    perpendicular_result = line_c.is_perpendicular_to(line_a)
    print(f'Are Line C and Line A perpendicular? {perpendicular_result}')
    
    circle_a = coordinate.create_circle(6, 3, 2)
    circle_area = circle_a.get_area()
    print(f'Print the area of Circle A. {circle_area}')
    
    circle_b = coordinate.create_circle(8, 1, 1)
    intersection_result = circle_a.intersects_with(circle_b)
    print(f'Do Circle A and Circle B intersect? {intersection_result}')
    
    polygon_points = [(-1, -2), (2, 0), (5, -1), (4, -4)]
    polygon_a = coordinate.create_polygon(polygon_points)
    perimeter = polygon_a.get_perimeter()
    print(f'Print the perimeter of Polygon A. {perimeter}')


def execute_task2():
    war_map = WarMap()
    
    war_map.add_enemy((-10, 2), (2, -1))  # E1
    war_map.add_enemy((-8, 0), (3, 1))    # E2
    war_map.add_enemy((-9, -1), (3, 0))   # E3
    
    war_map.add_basic_tower(-3, 2)   # T1
    war_map.add_basic_tower(-1, -2)  # T2
    war_map.add_basic_tower(4, 2)    # T3
    war_map.add_basic_tower(7, 0)    # T4
    
    war_map.add_advanced_tower(1, 1)   # A1
    war_map.add_advanced_tower(4, -3)  # A2
    
    game = Game(war_map)
    game.run(turns=10)

if __name__ == '__main__':
    execute_task1()
    print('====================')
    execute_task2()
