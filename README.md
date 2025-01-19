# we-help-deep-learning
Official repository for the WeHelp Deep Learning Camp 6th

## Week 2: Geometric Objects
This project implements geometric calculations including lines, circles, and polygons.

### Class Diagram
```mermaid
classDiagram
    class Point {
        +x: float
        +y: float
        +distance_to(target_point: Point) -> float
    }

    class Vector {
        +x: float
        +y: float
        +magnitude() -> float
        +normalize() -> Vector
        +dot_product(target_vector: Vector) -> float
        +is_parallel_to(target_vector: Vector) -> bool
    }

    class Line {
        +start: Point
        +end: Point
        +to_vector() -> Vector
        +is_parallel_to(target_line: Line) -> bool
        +is_perpendicular_to(target_line: Line) -> bool
    }

    class Circle {
        +center: Point
        +radius: float
        +get_area() -> float
        +intersects_with(target_circle: Circle) -> bool
    }

    class Polygon {
        +points: list[Point]
        +get_perimeter() -> float
    }

    class Coordinate {
        +create_point(x: float, y: float) -> Point
        +create_line(start_x: float, start_y: float, end_x: float, end_y: float) -> Line
        +create_circle(center_x: float, center_y: float, radius: float) -> Circle
        +create_polygon(points: list[tuple[float, float]]) -> Polygon
    }

    Line "1" *-- "2" Point
    Circle "1" *-- "1" Point
    Polygon "1" *-- "n" Point
    Line ..> Vector
    Coordinate ..> Point
    Coordinate ..> Line
    Coordinate ..> Circle
    Coordinate ..> Polygon
```

### Features
- Line parallel and perpendicular detection
- Circle area calculation and intersection detection
- Polygon perimeter calculation
