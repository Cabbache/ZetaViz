use rayon::prelude::*;
use rug::{Complex, Float};
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::mouse::MouseButton;
use sdl2::pixels::Color;
use sdl2::rect::Rect;
use sdl2::ttf::Font;
use std::time::Duration;

struct ViewState {
    re_min: f64,
    re_max: f64,
    im_min: f64,
    im_max: f64,
    precision: u32,
    compute_width: u32,
    compute_height: u32,
    show_axes: bool,
}

impl ViewState {
    fn new() -> Self {
        ViewState {
            re_min: -0.5,
            re_max: 1.5,
            im_min: -20.0,
            im_max: 20.0,
            precision: 64,
            compute_width: 160,
            compute_height: 120,
            show_axes: true,
        }
    }

    fn zoom(&mut self, factor: f64, center_re: f64, center_im: f64) {
        let re_range = self.re_max - self.re_min;
        let im_range = self.im_max - self.im_min;

        let new_re_range = re_range * factor;
        let new_im_range = im_range * factor;

        self.re_min = center_re - new_re_range * (center_re - self.re_min) / re_range;
        self.re_max = center_re + new_re_range * (self.re_max - center_re) / re_range;
        self.im_min = center_im - new_im_range * (center_im - self.im_min) / im_range;
        self.im_max = center_im + new_im_range * (self.im_max - center_im) / im_range;
    }

    fn pan(&mut self, delta_re: f64, delta_im: f64) {
        self.re_min += delta_re;
        self.re_max += delta_re;
        self.im_min += delta_im;
        self.im_max += delta_im;
    }

    fn pixel_to_complex(&self, x: i32, y: i32, window_width: u32, window_height: u32) -> (f64, f64) {
        let re = self.re_min + (x as f64 / window_width as f64) * (self.re_max - self.re_min);
        let im = self.im_max - (y as f64 / window_height as f64) * (self.im_max - self.im_min);
        (re, im)
    }
}

struct RenderState {
    buffer: Vec<Color>,
    current_row: u32,
    is_complete: bool,
}

impl RenderState {
    fn new(width: u32, height: u32) -> Self {
        RenderState {
            buffer: vec![Color::RGB(0, 0, 0); (width * height) as usize],
            current_row: 0,
            is_complete: false,
        }
    }

    fn reset(&mut self) {
        self.current_row = 0;
        self.is_complete = false;
        for pixel in &mut self.buffer {
            *pixel = Color::RGB(0, 0, 0);
        }
    }
}

fn main() {
    println!("Riemann Zeta Function Interactive Visualization");
    println!("=================================================");
    println!("Controls:");
    println!("  Mouse Drag: Pan view");
    println!("  Mouse Wheel: Zoom in/out");
    println!("  +/- : Increase/decrease precision");
    println!("  [/] : Decrease/increase computation resolution");
    println!("  A : Toggle axes");
    println!("  R : Reset view");
    println!("  ESC : Exit\n");

    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();
    let ttf_context = sdl2::ttf::init().unwrap();

    let window_width = 800u32;
    let window_height = 600u32;

    let window = video_subsystem
        .window("Riemann Zeta Visualization", window_width, window_height)
        .position_centered()
        .build()
        .unwrap();

    let mut canvas = window.into_canvas().build().unwrap();
    let mut event_pump = sdl_context.event_pump().unwrap();

    let font_paths = [
        "/usr/share/fonts/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ];

    let font = font_paths
        .iter()
        .find_map(|path| ttf_context.load_font(path, 14).ok())
        .expect("Could not load any font. Please install a basic TrueType font.");

    let mut view = ViewState::new();
    let mut render_state = RenderState::new(view.compute_width, view.compute_height);
    let mut needs_recompute = true;

    let mut is_dragging = false;
    let mut last_mouse_pos = (0i32, 0i32);
    let mut current_mouse_pos = (0i32, 0i32);

    let rows_per_frame = 20;

    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => break 'running,

                Event::MouseButtonDown {
                    mouse_btn: MouseButton::Left,
                    x,
                    y,
                    ..
                } => {
                    is_dragging = true;
                    last_mouse_pos = (x, y);
                }

                Event::MouseButtonUp {
                    mouse_btn: MouseButton::Left,
                    ..
                } => {
                    is_dragging = false;
                }

                Event::MouseMotion { x, y, .. } => {
                    current_mouse_pos = (x, y);
                    if is_dragging {
                        let dx = x - last_mouse_pos.0;
                        let dy = y - last_mouse_pos.1;

                        let re_range = view.re_max - view.re_min;
                        let im_range = view.im_max - view.im_min;

                        let delta_re = -(dx as f64 / window_width as f64) * re_range;
                        let delta_im = (dy as f64 / window_height as f64) * im_range;

                        view.pan(delta_re, delta_im);
                        needs_recompute = true;

                        last_mouse_pos = (x, y);
                    }
                }

                Event::MouseWheel { y: wheel_y, .. } => {
                    let (center_re, center_im) = view.pixel_to_complex(
                        current_mouse_pos.0,
                        current_mouse_pos.1,
                        window_width,
                        window_height,
                    );

                    let zoom_factor = if wheel_y > 0 { 0.8 } else { 1.25 };
                    view.zoom(zoom_factor, center_re, center_im);
                    needs_recompute = true;
                }

                Event::KeyDown {
                    keycode: Some(keycode),
                    ..
                } => match keycode {
                    Keycode::Equals | Keycode::Plus => {
                        view.precision = view.precision + 16;
                        println!("Precision: {}", view.precision);
                        needs_recompute = true;
                    }
                    Keycode::Minus => {
                        view.precision = view.precision.saturating_sub(16).max(16);
                        println!("Precision: {}", view.precision);
                        needs_recompute = true;
                    }
                    Keycode::LeftBracket => {
                        view.compute_width = (view.compute_width * 2 / 3).max(1);
                        view.compute_height = (view.compute_height * 2 / 3).max(1);
                        println!("Resolution: {}x{}", view.compute_width, view.compute_height);
                        render_state = RenderState::new(view.compute_width, view.compute_height);
                        needs_recompute = true;
                    }
                    Keycode::RightBracket => {
                        view.compute_width = (view.compute_width * 3 / 2).max(view.compute_width + 1);
                        view.compute_height = (view.compute_height * 3 / 2).max(view.compute_height + 1);
                        println!("Resolution: {}x{}", view.compute_width, view.compute_height);
                        render_state = RenderState::new(view.compute_width, view.compute_height);
                        needs_recompute = true;
                    }
                    Keycode::A => {
                        view.show_axes = !view.show_axes;
                        println!("Axes: {}", if view.show_axes { "ON" } else { "OFF" });
                    }
                    Keycode::R => {
                        view = ViewState::new();
                        render_state = RenderState::new(view.compute_width, view.compute_height);
                        needs_recompute = true;
                        println!("View reset");
                    }
                    _ => {}
                },

                _ => {}
            }
        }

        if needs_recompute {
            render_state.reset();
            needs_recompute = false;
            println!(
                "Computing: Re ∈ [{:.3}, {:.3}], Im ∈ [{:.3}, {:.3}], Prec: {}, Res: {}x{}",
                view.re_min, view.re_max, view.im_min, view.im_max, view.precision,
                view.compute_width, view.compute_height
            );
        }

        if !render_state.is_complete {
            let start_row = render_state.current_row;
            let end_row = (start_row + rows_per_frame).min(view.compute_height);

            let re_min = view.re_min;
            let re_max = view.re_max;
            let im_min = view.im_min;
            let im_max = view.im_max;
            let precision = view.precision;
            let compute_width = view.compute_width;
            let compute_height = view.compute_height;

            let rows: Vec<Vec<Color>> = (start_row..end_row)
                .into_par_iter()
                .map(|y| {
                    let mut row = Vec::with_capacity(compute_width as usize);
                    for x in 0..compute_width {
                        let re = re_min + (x as f64 / compute_width as f64) * (re_max - re_min);
                        let im = im_max - (y as f64 / compute_height as f64) * (im_max - im_min);

                        let s = Complex::with_val(precision, (re, im));
                        let zeta = riemann_zeta(&s);

                        let magnitude = (zeta.real().to_f64().powi(2) + zeta.imag().to_f64().powi(2)).sqrt();
                        let color = magnitude_to_color(magnitude);

                        row.push(color);
                    }
                    row
                })
                .collect();

            for (i, y) in (start_row..end_row).enumerate() {
                for (x, &color) in rows[i].iter().enumerate() {
                    render_state.buffer[(y * compute_width + x as u32) as usize] = color;
                }
            }

            render_state.current_row = end_row;
            if render_state.current_row >= view.compute_height {
                render_state.is_complete = true;
                println!("Rendering complete!");
            }
        }

        canvas.set_draw_color(Color::RGB(0, 0, 0));
        canvas.clear();

        for y in 0..view.compute_height {
            for x in 0..view.compute_width {
                let color = render_state.buffer[(y * view.compute_width + x) as usize];
                canvas.set_draw_color(color);

                let x1 = (x * window_width) / view.compute_width;
                let x2 = ((x + 1) * window_width) / view.compute_width;
                let y1 = (y * window_height) / view.compute_height;
                let y2 = ((y + 1) * window_height) / view.compute_height;

                let rect = Rect::new(
                    x1 as i32,
                    y1 as i32,
                    x2 - x1,
                    y2 - y1,
                );
                canvas.fill_rect(rect).unwrap();
            }
        }

        if view.show_axes {
            draw_axes(&mut canvas, &view, &font, window_width, window_height);
            draw_mouse_coordinates(&mut canvas, &view, &font, current_mouse_pos, window_width, window_height);
        }

        draw_info_overlay(&mut canvas, &view, &render_state, window_width, window_height);

        canvas.present();
        std::thread::sleep(Duration::from_millis(16));
    }
}

fn draw_axes(
    canvas: &mut sdl2::render::Canvas<sdl2::video::Window>,
    view: &ViewState,
    font: &Font,
    width: u32,
    height: u32,
) {
    canvas.set_draw_color(Color::RGBA(255, 255, 255, 180));

    let re_range = view.re_max - view.re_min;
    let im_range = view.im_max - view.im_min;

    let texture_creator = canvas.texture_creator();

    if view.re_min <= 0.0 && view.re_max >= 0.0 {
        let x = ((0.0 - view.re_min) / re_range * width as f64) as i32;
        for y in 0..height {
            canvas.draw_point((x, y as i32)).unwrap();
        }
    }

    if view.im_min <= 0.0 && view.im_max >= 0.0 {
        let y = ((view.im_max - 0.0) / im_range * height as f64) as i32;
        for x in 0..width {
            canvas.draw_point((x as i32, y)).unwrap();
        }
    }

    let num_ticks = 5;

    for i in 0..=num_ticks {
        let t = i as f64 / num_ticks as f64;
        let re_val = view.re_min + t * re_range;
        let im_val = view.im_max - t * im_range;

        let label_re = if re_val.abs() < 0.01 {
            "0".to_string()
        } else if re_val.abs() >= 1000.0 {
            format!("{:.1e}", re_val)
        } else {
            format!("{:.2}", re_val)
        };

        let label_im = if im_val.abs() < 0.01 {
            "0".to_string()
        } else if im_val.abs() >= 1000.0 {
            format!("{:.1e}i", im_val)
        } else {
            format!("{:.2}i", im_val)
        };

        if let Ok(surface) = font.render(&label_re).blended(Color::RGB(255, 255, 255)) {
            if let Ok(texture) = texture_creator.create_texture_from_surface(&surface) {
                let x_pos = (t * width as f64) as i32;
                let query = texture.query();
                let target = Rect::new(x_pos - query.width as i32 / 2, height as i32 - 20, query.width, query.height);

                let bg_rect = Rect::new(
                    target.x() - 2,
                    target.y() - 1,
                    target.width() + 4,
                    target.height() + 2,
                );
                canvas.set_draw_color(Color::RGBA(0, 0, 0, 180));
                canvas.fill_rect(bg_rect).ok();

                canvas.copy(&texture, None, target).ok();
            }
        }

        if let Ok(surface) = font.render(&label_im).blended(Color::RGB(255, 255, 255)) {
            if let Ok(texture) = texture_creator.create_texture_from_surface(&surface) {
                let y_pos = (t * height as f64) as i32;
                let query = texture.query();
                let target = Rect::new(5, y_pos - query.height as i32 / 2, query.width, query.height);

                let bg_rect = Rect::new(
                    target.x() - 2,
                    target.y() - 1,
                    target.width() + 4,
                    target.height() + 2,
                );
                canvas.set_draw_color(Color::RGBA(0, 0, 0, 180));
                canvas.fill_rect(bg_rect).ok();

                canvas.copy(&texture, None, target).ok();
            }
        }
    }
}

fn draw_mouse_coordinates(
    canvas: &mut sdl2::render::Canvas<sdl2::video::Window>,
    view: &ViewState,
    font: &Font,
    mouse_pos: (i32, i32),
    width: u32,
    height: u32,
) {
    let (mouse_x, mouse_y) = mouse_pos;

    if mouse_x < 0 || mouse_y < 0 || mouse_x >= width as i32 || mouse_y >= height as i32 {
        return;
    }

    let (re, im) = view.pixel_to_complex(mouse_x, mouse_y, width, height);

    let label = if re.abs() >= 1000.0 || im.abs() >= 1000.0 {
        format!("{:.2e} + {:.2e}i", re, im)
    } else {
        format!("{:.4} + {:.4}i", re, im)
    };

    let texture_creator = canvas.texture_creator();

    if let Ok(surface) = font.render(&label).blended(Color::RGB(255, 255, 255)) {
        if let Ok(texture) = texture_creator.create_texture_from_surface(&surface) {
            let query = texture.query();
            let padding = 6;
            let target = Rect::new(
                width as i32 - query.width as i32 - padding - 5,
                5,
                query.width,
                query.height,
            );

            let bg_rect = Rect::new(
                target.x() - padding / 2,
                target.y() - padding / 2,
                target.width() + padding as u32,
                target.height() + padding as u32,
            );
            canvas.set_draw_color(Color::RGBA(0, 0, 0, 200));
            canvas.fill_rect(bg_rect).ok();

            canvas.copy(&texture, None, target).ok();
        }
    }
}

fn draw_info_overlay(
    canvas: &mut sdl2::render::Canvas<sdl2::video::Window>,
    view: &ViewState,
    render_state: &RenderState,
    _width: u32,
    _height: u32,
) {
    let progress = if render_state.is_complete {
        100
    } else {
        (render_state.current_row as f32 / view.compute_height as f32 * 100.0) as u32
    };

    if !render_state.is_complete {
        canvas.set_draw_color(Color::RGBA(255, 255, 0, 200));
        let bar_height = 4;
        let bar_width = (800.0 * progress as f32 / 100.0) as u32;
        canvas
            .fill_rect(Rect::new(0, 0, bar_width, bar_height))
            .unwrap();
    }
}

fn magnitude_to_color(mag: f64) -> Color {
    let clamped = mag.min(10.0);

    if clamped < 0.1 {
        let intensity = (clamped / 0.1 * 255.0) as u8;
        Color::RGB(0, 0, intensity)
    } else if clamped < 1.0 {
        let t = (clamped - 0.1) / 0.9;
        let r = (t * 255.0) as u8;
        Color::RGB(r, 0, 255)
    } else if clamped < 3.0 {
        let t = (clamped - 1.0) / 2.0;
        let g = (t * 255.0) as u8;
        Color::RGB(255, g, (255.0 * (1.0 - t)) as u8)
    } else {
        Color::RGB(255, 255, 255)
    }
}

fn complex_pow(base: &Complex, exponent: &Complex) -> Complex {
    let ln_base = base.clone().ln();
    let product = exponent * ln_base;
    product.exp()
}

fn riemann_zeta(s: &Complex) -> Complex {
    let precision = s.prec().0;

    let one = Complex::with_val(precision, (1.0, 0.0));
    let s_minus_one = Complex::with_val(precision, s - &one);
    if s_minus_one.real().clone().abs() < 1e-10 && s_minus_one.imag().clone().abs() < 1e-10 {
        return Complex::with_val(precision, (Float::with_val(precision, 1e100), 0.0));
    }

    let eta = dirichlet_eta(s);

    let two = Complex::with_val(precision, (2.0, 0.0));
    let one_minus_s = Complex::with_val(precision, &one - s);
    let two_pow = complex_pow(&two, &one_minus_s);
    let denominator = &one - two_pow;

    eta / denominator
}

fn dirichlet_eta(s: &Complex) -> Complex {
    let precision = s.prec().0;
    let mut sum = Complex::with_val(precision, (0.0, 0.0));

    let max_terms = ((precision as f64 / 2.0).ceil() as usize).max(100).min(10000);

    for n in 1..=max_terms {
        let n_complex = Complex::with_val(precision, (n as f64, 0.0));
        let term = Complex::with_val(precision, (1.0, 0.0)) / complex_pow(&n_complex, s);

        if n % 2 == 1 {
            sum += &term;
        } else {
            sum -= &term;
        }

        let term_magnitude = term.real().clone().square() + term.imag().clone().square();
        let threshold = Float::with_val(precision, 1e-80);
        if term_magnitude < threshold {
            break;
        }
    }

    sum
}
