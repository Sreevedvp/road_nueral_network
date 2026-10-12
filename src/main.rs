use macroquad::prelude::*;

const WIDTH: usize = 1000;
const HEIGHT: usize = 700;
const POPULATION: usize = 40;
const BRUSH: i32 = 35;
const CAR_RADIUS: f32 = 6.0;
const DT: f32 = 1.0 / 60.0;

const INPUTS: usize = 11; // 8 sensors + 2 direction values + 1 distance
const HIDDEN: usize = 8;
const OUTPUTS: usize = 2; // steering and speed

const WEIGHTS: usize = HIDDEN * (INPUTS + 1) + OUTPUTS * (HIDDEN + 1);

// Angles relative to the car's forward direction, in radians.
// Negative = left, positive = right.
const SENSOR_ANGLES: [f32; 8] = [-1.40, -1.00, -0.60, -0.20, 0.20, 0.60, 1.00, 1.40];

const SENSOR_RANGE: f32 = 160.0;

fn config() -> Conf {
    Conf {
        window_title: "Road AI".to_owned(),
        window_width: WIDTH as i32,
        window_height: HEIGHT as i32,
        window_resizable: false,
        ..Default::default()
    }
}

fn direction(angle: f32) -> Vec2 {
    vec2(angle.cos(), angle.sin())
}

#[derive(Clone)]
struct Brain {
    weights: [f32; WEIGHTS],
}
#[derive(Clone)]
struct Example {
    inputs: [f32; INPUTS],
    steering: f32,
    throttle: f32,
}
impl Brain {
    fn random() -> Self {
        let mut weights = [0.0; WEIGHTS];
        for weight in &mut weights {
            *weight = rand::gen_range(-1.0, 1.0);
        }
        Self { weights }
    }

    fn drive(&self, inputs: [f32; INPUTS]) -> (f32, f32) {
        let mut hidden = [0.0; HIDDEN];
        for neuron in 0..HIDDEN {
            let offset = neuron * (INPUTS + 1);
            let mut total = self.weights[offset + INPUTS];
            for input in 0..INPUTS {
                total += inputs[input] * self.weights[offset + input];
            }
            hidden[neuron] = total.tanh();
        }

        let output_start = HIDDEN * (INPUTS + 1);
        let mut outputs = [0.0; OUTPUTS];
        for output in 0..OUTPUTS {
            let offset = output_start + output * (HIDDEN + 1);
            let mut total = self.weights[offset + HIDDEN];
            for neuron in 0..HIDDEN {
                total += hidden[neuron] * self.weights[offset + neuron];
            }
            outputs[output] = total.tanh();
        }
        (outputs[0], (outputs[1] + 1.0) / 2.0)
    }

    fn mutated(&self) -> Self {
        let mut child = self.clone();
        for weight in &mut child.weights {
            if rand::gen_range(0.0, 1.0) < 0.12 {
                *weight = (*weight + rand::gen_range(-0.20, 0.20)).clamp(-5.0, 5.0);
            }
        }
        child
    }
}

impl Brain {
    fn learn(&mut self, example: &Example, learning_rate: f32) {
        // Forward pass through the hidden layer.
        let mut hidden = [0.0; HIDDEN];

        for j in 0..HIDDEN {
            let offset = j * (INPUTS + 1);
            let mut total = self.weights[offset + INPUTS];

            for i in 0..INPUTS {
                total += self.weights[offset + i] * example.inputs[i];
            }

            hidden[j] = total.tanh();
        }

        let (steering, throttle) = self.drive(example.inputs);
        let raw_throttle = throttle * 2.0 - 1.0;

        // Derivatives of:
        // (steering - target)^2 + (throttle - target)^2
        //
        // Throttle includes the transformation (tanh(z) + 1) / 2.
        let output_delta = [
            2.0 * (steering - example.steering) * (1.0 - steering * steering),
            (throttle - example.throttle) * (1.0 - raw_throttle * raw_throttle),
        ];

        let output_start = HIDDEN * (INPUTS + 1);

        // Calculate hidden gradients BEFORE changing output weights.
        let mut hidden_delta = [0.0; HIDDEN];

        for j in 0..HIDDEN {
            let mut error = 0.0;

            for k in 0..OUTPUTS {
                let offset = output_start + k * (HIDDEN + 1);
                error += output_delta[k] * self.weights[offset + j];
            }

            hidden_delta[j] = error * (1.0 - hidden[j] * hidden[j]);
        }

        // Update output-layer weights and biases.
        for k in 0..OUTPUTS {
            let offset = output_start + k * (HIDDEN + 1);

            for j in 0..HIDDEN {
                self.weights[offset + j] -= learning_rate * output_delta[k] * hidden[j];
            }

            self.weights[offset + HIDDEN] -= learning_rate * output_delta[k];
        }

        // Update hidden-layer weights and biases.
        for j in 0..HIDDEN {
            let offset = j * (INPUTS + 1);

            for i in 0..INPUTS {
                self.weights[offset + i] -= learning_rate * hidden_delta[j] * example.inputs[i];
            }

            self.weights[offset + INPUTS] -= learning_rate * hidden_delta[j];
        }
    }
}

struct Road {
    image: Image,
    texture: Texture2D,
}

impl Road {
    fn new() -> Self {
        let image = Image::gen_image_color(WIDTH as u16, HEIGHT as u16, WHITE);
        let texture = Texture2D::from_image(&image);
        Self { image, texture }
    }

    fn contains(&self, point: Vec2) -> bool {
        if point.x < 0.0 || point.y < 90.0 || point.x >= WIDTH as f32 || point.y >= HEIGHT as f32 {
            return false;
        }

        self.image.get_pixel(point.x as u32, point.y as u32).r < 0.5
    }

    fn fits_car(&self, position: Vec2) -> bool {
        self.contains(position)
            && (0..12).all(|i| {
                let angle = i as f32 * std::f32::consts::TAU / 12.0;
                self.contains(position + direction(angle) * CAR_RADIUS)
            })
    }

    fn paint(&mut self, from: Vec2, to: Vec2) {
        let steps = (from.distance(to) / 4.0).ceil().max(1.0) as usize;

        for step in 0..=steps {
            let center = from.lerp(to, step as f32 / steps as f32);

            for dy in -BRUSH..=BRUSH {
                for dx in -BRUSH..=BRUSH {
                    if dx * dx + dy * dy > BRUSH * BRUSH {
                        continue;
                    }

                    let x = center.x as i32 + dx;
                    let y = center.y as i32 + dy;

                    if x >= 0 && x < WIDTH as i32 && y >= 90 && y < HEIGHT as i32 {
                        self.image.set_pixel(x as u32, y as u32, BLACK);
                    }
                }
            }
        }

        self.texture.update(&self.image);
    }

    fn sensor(&self, position: Vec2, angle: f32) -> f32 {
        let step_size = 2.0;
        let steps = (SENSOR_RANGE / step_size) as usize;

        for step in 1..=steps {
            let distance = step as f32 * step_size;
            let point = position + direction(angle) * distance;

            if !self.contains(point) {
                return distance / SENSOR_RANGE;
            }
        }

        1.0
    }
}

#[derive(Clone)]
struct Car {
    brain: Brain,
    position: Vec2,
    angle: f32,
    next_checkpoint: usize,
    elapsed: f32,
    since_checkpoint: f32,
    score: f32,
    alive: bool,
    finished: bool,
}

impl Car {
    fn new(brain: Brain, route: &[Vec2]) -> Self {
        let heading = route[1] - route[0];

        Self {
            brain,
            position: route[0],
            angle: heading.y.atan2(heading.x),
            next_checkpoint: 1,
            elapsed: 0.0,
            since_checkpoint: 0.0,
            score: 0.0,
            alive: true,
            finished: false,
        }
    }

    fn observe(&self, road: &Road, route: &[Vec2]) -> [f32; INPUTS] {
        let displacement = route[self.next_checkpoint] - self.position;
        let distance = displacement.length();
        let target_direction = displacement.normalize_or_zero();
        let forward = direction(self.angle);
        let right = vec2(-forward.y, forward.x);
        let mut inputs = [0.0; INPUTS];

        for i in 0..8 {
            inputs[i] = road.sensor(self.position, self.angle + SENSOR_ANGLES[i]);
        }
        inputs[8] = forward.dot(target_direction);
        inputs[9] = right.dot(target_direction);
        inputs[10] = distance / (distance + 150.0);
        inputs
    }

    fn update(&mut self, road: &Road, route: &[Vec2], manual: Option<(f32, f32)>) {
        if !self.alive {
            return;
        }

        self.elapsed += DT;
        self.since_checkpoint += DT;

        let target = route[self.next_checkpoint];
        let distance_to_target = self.position.distance(target);
        let inputs = self.observe(road, route);

        let (steering, throttle) = match manual {
            Some(controls) => controls,
            None => self.brain.drive(inputs),
        };

        let steering = steering.clamp(-1.0, 1.0);
        let throttle = throttle.clamp(0.0, 1.0);

        let requested_speed = throttle * 120.0;

        let front_clearance = inputs[3].min(inputs[4]) * SENSOR_RANGE;

        let clearance_factor = ((front_clearance - CAR_RADIUS) / 80.0).clamp(0.2, 1.0);

        let turning_factor = 1.0 - 0.55 * steering.abs();

        let checkpoint_factor = (distance_to_target / 70.0).clamp(0.4, 1.0);

        let speed = (requested_speed * clearance_factor * turning_factor * checkpoint_factor)
            .clamp(0.0, 120.0);

        let curvature = steering * 0.045;
        let travel_distance = speed * DT;

        let substeps = (travel_distance / 2.0).ceil().max(1.0) as usize;

        let step_distance = travel_distance / substeps as f32;

        for _ in 0..substeps {
            let heading_change = curvature * step_distance;
            let midpoint_angle = self.angle + heading_change * 0.5;

            let next_position = self.position + direction(midpoint_angle) * step_distance;

            if !road.fits_car(next_position) {
                self.alive = false;
                return;
            }

            self.position = next_position;
            self.angle += heading_change;

            if self.position.distance(route[self.next_checkpoint]) < 18.0 {
                self.next_checkpoint += 1;
                self.since_checkpoint = 0.0;

                if self.next_checkpoint == route.len() {
                    self.finished = true;
                    self.alive = false;

                    self.score =
                        route.len() as f32 * 1000.0 + 10000.0 + (1000.0 - self.elapsed).max(0.0);

                    return;
                }

                break;
            }
        }

        let previous = route[self.next_checkpoint - 1];
        let next = route[self.next_checkpoint];
        let length = previous.distance(next).max(1.0);

        let progress = (1.0 - self.position.distance(next) / length).clamp(0.0, 1.0);

        let score = (self.next_checkpoint - 1) as f32 * 1000.0 + progress * 900.0;

        self.score = self.score.max(score);

        // Let a human pause to think without timing out.
        // AI attempts still have time limits.
        if manual.is_none() && (self.since_checkpoint > 15.0 || self.elapsed > 120.0) {
            self.alive = false;
        }
    }
}

#[macroquad::main(config)]
async fn main() {
    rand::srand(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    );

    let mut road = Road::new();
    let mut route: Vec<Vec2> = Vec::new();
    let mut previous_mouse: Option<Vec2> = None;

    let mut cars: Vec<Car> = Vec::new();
    let mut best: Option<Brain> = None;
    let mut best_score = f32::NEG_INFINITY;
    let mut generation = 0;

    // 0 = edit, 1 = train, 2 = test.
    let mut mode = 0;
    let mut accumulator = 0.0;
    let mut message = String::from("Draw a road, then right-click route points.");

    loop {
        if is_key_pressed(KeyCode::Escape) {
            break;
        }

        let (mx, my) = mouse_position();
        let mouse = vec2(mx, my);

        // Return to editing while keeping the learned brain.
        if is_key_pressed(KeyCode::E) {
            mode = 0;
            cars.clear();
            previous_mouse = None;
            best_score = f32::NEG_INFINITY;
            message = "Editing. Best brain kept in memory.".to_owned();
        }

        // Clear the map, but keep the learned brain.
        if is_key_pressed(KeyCode::C) {
            mode = 0;
            road = Road::new();
            route.clear();
            cars.clear();
            previous_mouse = None;
            best_score = f32::NEG_INFINITY;
            message = "New map. Best brain kept. Draw and place route points.".to_owned();
        }

        if mode == 0 {
            if is_mouse_button_down(MouseButton::Left) && mouse.y >= 90.0 {
                road.paint(previous_mouse.unwrap_or(mouse), mouse);
                previous_mouse = Some(mouse);
                best_score = f32::NEG_INFINITY;
            } else {
                previous_mouse = None;
            }

            if is_mouse_button_pressed(MouseButton::Right) {
                let separated = route
                    .last()
                    .map_or(true, |point| point.distance(mouse) >= 30.0);

                if road.fits_car(mouse) && separated {
                    route.push(mouse);
                    best_score = f32::NEG_INFINITY;
                } else {
                    message = "Place points inside the road, at least 30px apart.".to_owned();
                }
            }

            if is_key_pressed(KeyCode::Backspace) {
                route.pop();
                best_score = f32::NEG_INFINITY;
            }
        }

        let train_pressed = is_key_pressed(KeyCode::T);
        let test_pressed = is_key_pressed(KeyCode::V);

        if mode == 0 && (train_pressed || test_pressed) {
            if route.len() < 2 {
                message = "Right-click a start, checkpoints, and finish first.".to_owned();
            } else if test_pressed {
                if let Some(brain) = &best {
                    cars = vec![Car::new(brain.clone(), &route)];
                    mode = 2;
                    accumulator = 0.0;
                    message = "Testing: brain weights are frozen.".to_owned();
                } else {
                    message = "Train with T before testing.".to_owned();
                }
            } else {
                let seed = best.clone().unwrap_or_else(Brain::random);

                cars = (0..POPULATION)
                    .map(|i| {
                        let brain = if i == 0 {
                            seed.clone()
                        } else if i < POPULATION / 2 {
                            seed.mutated()
                        } else {
                            Brain::random()
                        };

                        Car::new(brain, &route)
                    })
                    .collect();

                generation = 1;
                best_score = f32::NEG_INFINITY;
                mode = 1;
                accumulator = 0.0;
                message = "Training. Hold Space to speed up.".to_owned();
            }
        }

        if mode != 0 {
            let speed = if mode == 1 && is_key_down(KeyCode::Space) {
                8.0
            } else {
                1.0
            };

            accumulator += get_frame_time().min(0.05) * speed;

            while accumulator >= DT {
                accumulator -= DT;

                for car in &mut cars {
                    car.update(&road, &route, None);
                }

                if mode == 1 {
                    for car in &cars {
                        if car.score > best_score {
                            best_score = car.score;
                            best = Some(car.brain.clone());
                        }
                    }

                    if cars.iter().all(|car| !car.alive) {
                        cars.sort_by(|a, b| b.score.total_cmp(&a.score));

                        let parents: Vec<Brain> =
                            cars.iter().take(8).map(|car| car.brain.clone()).collect();

                        let champion = best.as_ref().unwrap().clone();

                        cars = (0..POPULATION)
                            .map(|i| {
                                let brain = if i == 0 {
                                    champion.clone()
                                } else if i <= parents.len() {
                                    parents[i - 1].clone()
                                } else if i >= POPULATION - 2 {
                                    Brain::random()
                                } else {
                                    let parent = rand::gen_range(0, parents.len());
                                    parents[parent].mutated()
                                };

                                Car::new(brain, &route)
                            })
                            .collect();

                        generation += 1;
                    }
                } else if !cars[0].alive {
                    message = if cars[0].finished {
                        "Test complete: reached the finish! E: edit".to_owned()
                    } else {
                        "Test ended: crashed or timed out. E: edit".to_owned()
                    };

                    accumulator = 0.0;
                    break;
                }
            }
        } else {
            accumulator = 0.0;
        }

        clear_background(WHITE);
        draw_texture(&road.texture, 0.0, 0.0, WHITE);

        for (i, point) in route.iter().enumerate() {
            let color = if i == 0 {
                GREEN
            } else if i == route.len() - 1 {
                RED
            } else {
                YELLOW
            };

            draw_circle(point.x, point.y, 10.0, color);
            draw_text(&i.to_string(), point.x + 12.0, point.y, 20.0, BLUE);
        }

        for car in &cars {
            let color = if car.finished {
                GREEN
            } else if car.alive {
                SKYBLUE
            } else {
                GRAY
            };

            draw_circle(car.position.x, car.position.y, CAR_RADIUS, color);

            let nose = car.position + direction(car.angle) * 11.0;
            draw_line(car.position.x, car.position.y, nose.x, nose.y, 2.0, ORANGE);
        }

        draw_rectangle(0.0, 0.0, WIDTH as f32, 90.0, LIGHTGRAY);
        draw_text(
            "Left drag: road | Right click: route | Backspace: undo point",
            12.0,
            23.0,
            21.0,
            BLACK,
        );
        draw_text(
            "T: train | V: test best | E: edit | C: new map | Hold Space: faster",
            12.0,
            46.0,
            21.0,
            BLACK,
        );

        let status = if mode == 1 {
            format!(
                "Generation {} | Best {:.0} | Alive {}",
                generation,
                best_score.max(0.0),
                cars.iter().filter(|car| car.alive).count(),
            )
        } else {
            message.clone()
        };

        draw_text(&status, 12.0, 73.0, 21.0, DARKBLUE);
        next_frame().await;
    }
}
