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
    log_scale: bool,
    filter_re_ge_1: bool,
}

impl ViewState {
    fn new() -> Self {
        ViewState {
            re_min: -19.5,
            re_max: 20.5,
            im_min: -15.0,
            im_max: 15.0,
            precision: 64,
            compute_width: 160,
            compute_height: 120,
            show_axes: true,
            log_scale: true,
            filter_re_ge_1: false,
        }
    }

    fn zoom(
        &mut self,
        factor: f64,
        center_re: f64,
        center_im: f64,
        window_width: u32,
        window_height: u32,
    ) {
        let re_range = self.re_max - self.re_min;
        let im_range = self.im_max - self.im_min;

        let new_re_range = re_range * factor;
        let new_im_range = im_range * factor;

        self.re_min = center_re - new_re_range * (center_re - self.re_min) / re_range;
        self.re_max = center_re + new_re_range * (self.re_max - center_re) / re_range;
        self.im_min = center_im - new_im_range * (center_im - self.im_min) / im_range;
        self.im_max = center_im + new_im_range * (self.im_max - center_im) / im_range;

        self.correct_aspect_ratio(window_width, window_height);
    }

    fn correct_aspect_ratio(&mut self, window_width: u32, window_height: u32) {
        let current_re_range = self.re_max - self.re_min;
        let current_im_range = self.im_max - self.im_min;
        let current_ratio = current_re_range / current_im_range;
        let target_ratio = window_width as f64 / window_height as f64;

        if (current_ratio - target_ratio).abs() > 0.001 {
            let center_re = (self.re_min + self.re_max) / 2.0;

            let new_re_range = current_im_range * target_ratio;
            self.re_min = center_re - new_re_range / 2.0;
            self.re_max = center_re + new_re_range / 2.0;
        }
    }

    fn pan(&mut self, delta_re: f64, delta_im: f64) {
        self.re_min += delta_re;
        self.re_max += delta_re;
        self.im_min += delta_im;
        self.im_max += delta_im;
        // Don't correct aspect ratio during pan - it breaks incremental rendering
    }

    fn pixel_to_complex(
        &self,
        x: i32,
        y: i32,
        window_width: u32,
        window_height: u32,
    ) -> (f64, f64) {
        let re = self.re_min + (x as f64 / window_width as f64) * (self.re_max - self.re_min);
        let im = self.im_max - (y as f64 / window_height as f64) * (self.im_max - self.im_min);
        (re, im)
    }
}

struct RenderState {
    buffer: Vec<Color>,
    current_row: u32,
    is_complete: bool,
    prev_re_min: f64,
    prev_re_max: f64,
    prev_im_min: f64,
    prev_im_max: f64,
    prev_precision: u32,
    prev_log_scale: bool,
    regions_to_compute: Vec<(u32, u32, u32, u32)>, // (x_start, y_start, x_end, y_end)
    current_region_idx: usize,
    current_region_row: u32,
    // Frozen view bounds for rendering (to avoid coordinate drift during panning)
    render_re_min: f64,
    render_re_max: f64,
    render_im_min: f64,
    render_im_max: f64,
}

impl RenderState {
    fn new(view: &ViewState, width: u32, height: u32) -> Self {
        RenderState {
            buffer: vec![Color::RGB(0, 0, 0); (width * height) as usize],
            current_row: 0,
            is_complete: false,
            prev_re_min: 0.0,
            prev_re_max: 0.0,
            prev_im_min: 0.0,
            prev_im_max: 0.0,
            prev_precision: 0, // Set to 0 to force full recompute on first render
            prev_log_scale: view.log_scale,
            regions_to_compute: vec![(0, 0, width, height)],
            current_region_idx: 0,
            current_region_row: 0,
            render_re_min: view.re_min,
            render_re_max: view.re_max,
            render_im_min: view.im_min,
            render_im_max: view.im_max,
        }
    }

    fn start_full_recompute(&mut self, view: &ViewState, width: u32, height: u32) {
        self.current_row = 0;
        self.is_complete = false;
        self.regions_to_compute = vec![(0, 0, width, height)];
        self.current_region_idx = 0;
        self.current_region_row = 0;
        for pixel in &mut self.buffer {
            *pixel = Color::RGB(0, 0, 0);
        }

        // Update all prev values
        self.prev_re_min = view.re_min;
        self.prev_re_max = view.re_max;
        self.prev_im_min = view.im_min;
        self.prev_im_max = view.im_max;
        self.prev_precision = view.precision;
        self.prev_log_scale = view.log_scale;

        // Freeze render coordinates
        self.render_re_min = view.re_min;
        self.render_re_max = view.re_max;
        self.render_im_min = view.im_min;
        self.render_im_max = view.im_max;
    }

    fn start_incremental_recompute(&mut self, view: &mut ViewState, width: u32, height: u32) {
        let old_view = (
            self.prev_re_min,
            self.prev_re_max,
            self.prev_im_min,
            self.prev_im_max,
        );

        self.regions_to_compute = self.shift_and_fill(old_view, view, width, height);
        self.current_region_idx = 0;
        self.current_region_row = if !self.regions_to_compute.is_empty() {
            self.regions_to_compute[0].1 // y_start of first region
        } else {
            0
        };

        // If no regions to compute, we're done
        self.is_complete = self.regions_to_compute.is_empty();

        // Update prev values immediately so next pan can be incremental too
        self.prev_re_min = view.re_min;
        self.prev_re_max = view.re_max;
        self.prev_im_min = view.im_min;
        self.prev_im_max = view.im_max;

        // CRITICAL: Freeze render coordinates after snapping
        // These will be used for computing pixels and must not change during rendering
        self.render_re_min = view.re_min;
        self.render_re_max = view.re_max;
        self.render_im_min = view.im_min;
        self.render_im_max = view.im_max;
    }

    fn can_incremental_update(&self, view: &ViewState, width: u32, height: u32) -> bool {
        // Can do incremental update if precision, resolution have not changed.
        // CRITICAL FIX: We must also check if the *scale* (zoom level) has changed.
        // If we zoomed, the pixel/complex ratio changed, so we cannot shift pixels.
        // Panning preserves the range, Zooming changes it.

        let prev_re_range = self.prev_re_max - self.prev_re_min;
        let prev_im_range = self.prev_im_max - self.prev_im_min;
        let new_re_range = view.re_max - view.re_min;
        let new_im_range = view.im_max - view.im_min;

        // Use a small epsilon for floating point comparison
        let range_preserved = (new_re_range / prev_re_range - 1.0).abs() < 1e-6
            && (new_im_range / prev_im_range - 1.0).abs() < 1e-6;

        self.prev_precision != 0  // Has been initialized
            && self.prev_precision == view.precision
            && self.prev_log_scale == view.log_scale
            && self.buffer.len() == (width * height) as usize
            && range_preserved
    }

    fn shift_and_fill(
        &mut self,
        old_view: (f64, f64, f64, f64),
        new_view: &mut ViewState,
        width: u32,
        height: u32,
    ) -> Vec<(u32, u32, u32, u32)> {
        // Returns regions that need to be recomputed: (x_start, y_start, x_end, y_end)
        let (old_re_min, _old_re_max, _old_im_min, old_im_max) = old_view;

        let new_re_range = new_view.re_max - new_view.re_min;
        let new_im_range = new_view.im_max - new_view.im_min;

        // Calculate pixel shift (exact floating point)
        let dx_exact = (old_re_min - new_view.re_min) / new_re_range * width as f64;
        let dy_exact = (new_view.im_max - old_im_max) / new_im_range * height as f64;

        // Round to nearest integer
        let dx_pixels = dx_exact.round() as i32;
        let dy_pixels = dy_exact.round() as i32;

        // Check for sub-pixel rounding errors
        let dx_error = (dx_exact - dx_pixels as f64).abs();
        let dy_error = (dy_exact - dy_pixels as f64).abs();
        let max_error = dx_error.max(dy_error);

        // Snap view coordinates to pixel boundaries to eliminate sub-pixel errors
        if max_error > 0.01 {
            let re_per_pixel = new_re_range / width as f64;
            let im_per_pixel = new_im_range / height as f64;

            // Adjust view to match the rounded pixel shift
            let re_correction = (dx_exact - dx_pixels as f64) * re_per_pixel;
            let im_correction = (dy_exact - dy_pixels as f64) * im_per_pixel;

            new_view.re_min += re_correction;
            new_view.re_max += re_correction;
            new_view.im_min -= im_correction; // Note: im is inverted in screen coords
            new_view.im_max -= im_correction;
        }

        // If shift is too large, just recompute everything
        if dx_pixels.abs() >= width as i32 || dy_pixels.abs() >= height as i32 {
            return vec![(0, 0, width, height)];
        }

        // If no shift at all, nothing to do
        if dx_pixels == 0 && dy_pixels == 0 {
            return vec![];
        }

        // Create new buffer
        let mut new_buffer = vec![Color::RGB(0, 0, 0); (width * height) as usize];

        // Copy shifted pixels
        for y in 0..height as i32 {
            for x in 0..width as i32 {
                let old_x = x - dx_pixels;
                let old_y = y - dy_pixels;

                if old_x >= 0 && old_x < width as i32 && old_y >= 0 && old_y < height as i32 {
                    let old_idx = (old_y * width as i32 + old_x) as usize;
                    let new_idx = (y * width as i32 + x) as usize;
                    new_buffer[new_idx] = self.buffer[old_idx];
                }
            }
        }

        self.buffer = new_buffer;

        // Determine regions to recompute
        let mut regions = Vec::new();

        if dx_pixels > 0 {
            // Need to fill left strip
            regions.push((0, 0, dx_pixels as u32, height));
        } else if dx_pixels < 0 {
            // Need to fill right strip
            let x_start = (width as i32 + dx_pixels) as u32;
            regions.push((x_start, 0, width, height));
        }

        if dy_pixels > 0 {
            // Need to fill top strip (excluding already computed horizontal strip)
            let x_start = if dx_pixels > 0 { dx_pixels as u32 } else { 0 };
            let x_end = if dx_pixels < 0 {
                (width as i32 + dx_pixels) as u32
            } else {
                width
            };
            regions.push((x_start, 0, x_end, dy_pixels as u32));
        } else if dy_pixels < 0 {
            // Need to fill bottom strip (excluding already computed horizontal strip)
            let x_start = if dx_pixels > 0 { dx_pixels as u32 } else { 0 };
            let x_end = if dx_pixels < 0 {
                (width as i32 + dx_pixels) as u32
            } else {
                width
            };
            let y_start = (height as i32 + dy_pixels) as u32;
            regions.push((x_start, y_start, x_end, height));
        }

        regions
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
    println!("  L : Toggle logarithmic color scale");
    println!("  F : Toggle filter (hide Re >= 1)");
    println!("  R : Reset view");
    println!("  Space : Force re-render");
    println!("  ESC : Exit\n");

    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();
    let ttf_context = sdl2::ttf::init().unwrap();

    let window_width = 1920u32;
    let window_height = 1080u32;

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
    let mut render_state = RenderState::new(&view, view.compute_width, view.compute_height);
    let mut needs_recompute = true;
    let mut pending_recompute = false; // Set when we get pan during rendering

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

                        if render_state.is_complete {
                            needs_recompute = true;
                        } else {
                            pending_recompute = true;
                        }

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
                    view.zoom(
                        zoom_factor,
                        center_re,
                        center_im,
                        window_width,
                        window_height,
                    );
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
                        render_state =
                            RenderState::new(&view, view.compute_width, view.compute_height);
                        needs_recompute = true;
                    }
                    Keycode::RightBracket => {
                        view.compute_width =
                            (view.compute_width * 3 / 2).max(view.compute_width + 1);
                        view.compute_height =
                            (view.compute_height * 3 / 2).max(view.compute_height + 1);
                        println!("Resolution: {}x{}", view.compute_width, view.compute_height);
                        render_state =
                            RenderState::new(&view, view.compute_width, view.compute_height);
                        needs_recompute = true;
                    }
                    Keycode::A => {
                        view.show_axes = !view.show_axes;
                        println!("Axes: {}", if view.show_axes { "ON" } else { "OFF" });
                    }
                    Keycode::L => {
                        view.log_scale = !view.log_scale;
                        println!(
                            "Logarithmic scale: {}",
                            if view.log_scale { "ON" } else { "OFF" }
                        );
                        needs_recompute = true;
                    }
                    Keycode::R => {
                        view = ViewState::new();
                        render_state =
                            RenderState::new(&view, view.compute_width, view.compute_height);
                        needs_recompute = true;
                        println!("View reset");
                    }
                    Keycode::Space => {
                        render_state =
                            RenderState::new(&view, view.compute_width, view.compute_height);
                        needs_recompute = true;
                        println!("Force re-render");
                    }
                    Keycode::F => {
                        view.filter_re_ge_1 = !view.filter_re_ge_1;
                        println!(
                            "Hide Re >= 1: {}",
                            if view.filter_re_ge_1 { "ON" } else { "OFF" }
                        );
                        needs_recompute = true;
                    }
                    _ => {}
                },

                _ => {}
            }
        }

        if needs_recompute {
            let can_incremental =
                render_state.can_incremental_update(&view, view.compute_width, view.compute_height);

            if can_incremental {
                let width = view.compute_width;
                let height = view.compute_height;
                render_state.start_incremental_recompute(&mut view, width, height);
            } else {
                render_state.start_full_recompute(&view, view.compute_width, view.compute_height);
            }
            needs_recompute = false;
        }

        if !render_state.is_complete {
            // CRITICAL: Use frozen render coordinates, not current view coordinates
            // This prevents artifacts when panning during incremental rendering
            let re_min = render_state.render_re_min;
            let re_max = render_state.render_re_max;
            let im_min = render_state.render_im_min;
            let im_max = render_state.render_im_max;
            let precision = view.precision;
            let compute_width = view.compute_width;
            let compute_height = view.compute_height;
            let log_scale = view.log_scale;
            let filter_re_ge_1 = view.filter_re_ge_1;

            // Process current region
            if render_state.current_region_idx < render_state.regions_to_compute.len() {
                let (x_start, _y_start, x_end, y_end) =
                    render_state.regions_to_compute[render_state.current_region_idx];

                let start_row = render_state.current_region_row;
                let end_row = (start_row + rows_per_frame).min(y_end);

                // Compute pixels in the current region
                let pixels: Vec<(u32, u32, Color)> = (start_row..end_row)
                    .into_par_iter()
                    .flat_map(|y| {
                        let mut row_pixels = Vec::new();
                        for x in x_start..x_end {
                            let re = re_min + (x as f64 / compute_width as f64) * (re_max - re_min);
                            let im =
                                im_max - (y as f64 / compute_height as f64) * (im_max - im_min);

                            let color = if filter_re_ge_1 && re >= 1.0 {
                                // Skip computation for Re >= 1 when filter is enabled
                                Color::RGB(0, 0, 0)
                            } else {
                                let s = Complex::with_val(precision, (re, im));
                                let zeta = riemann_zeta(&s);

                                let magnitude = (zeta.real().to_f64().powi(2)
                                    + zeta.imag().to_f64().powi(2))
                                .sqrt();
                                magnitude_to_color(magnitude, log_scale)
                            };

                            row_pixels.push((x, y, color));
                        }
                        row_pixels
                    })
                    .collect();

                // Write pixels to buffer
                for (x, y, color) in pixels {
                    render_state.buffer[(y * compute_width + x) as usize] = color;
                }

                render_state.current_region_row = end_row;

                // Move to next region if current region is complete
                if render_state.current_region_row >= y_end {
                    render_state.current_region_idx += 1;
                    render_state.current_region_row = if render_state.current_region_idx
                        < render_state.regions_to_compute.len()
                    {
                        render_state.regions_to_compute[render_state.current_region_idx].1
                    } else {
                        0
                    };
                }

                // Check if all regions are complete
                if render_state.current_region_idx >= render_state.regions_to_compute.len() {
                    render_state.is_complete = true;

                    // If we accumulated pans during rendering, process them now
                    if pending_recompute {
                        needs_recompute = true;
                        pending_recompute = false;
                    }
                }
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

                let rect = Rect::new(x1 as i32, y1 as i32, x2 - x1, y2 - y1);
                canvas.fill_rect(rect).unwrap();
            }
        }

        if view.show_axes {
            draw_axes(&mut canvas, &view, &font, window_width, window_height);
            draw_mouse_coordinates(
                &mut canvas,
                &view,
                &font,
                current_mouse_pos,
                window_width,
                window_height,
            );
        }

        draw_info_overlay(
            &mut canvas,
            &view,
            &render_state,
            window_width,
            window_height,
        );

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
                let target = Rect::new(
                    x_pos - query.width as i32 / 2,
                    height as i32 - 20,
                    query.width,
                    query.height,
                );

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
                let target = Rect::new(
                    5,
                    y_pos - query.height as i32 / 2,
                    query.width,
                    query.height,
                );

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
    width: u32,
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
        let bar_width = (width as f32 * progress as f32 / 100.0) as u32;
        canvas
            .fill_rect(Rect::new(0, 0, bar_width, bar_height))
            .unwrap();
    }
}

fn magnitude_to_color(mag: f64, use_log_scale: bool) -> Color {
    if use_log_scale {
        // Logarithmic scale - better for showing zeros
        let log_mag = if mag < 1e-10 {
            -10.0
        } else {
            mag.log10().max(-10.0).min(1.0)
        };

        // Map log_mag from [-10, 1] to colors
        // -10 to -2: black to dark blue (very dark for zeros)
        // -2 to 0: dark blue to bright blue
        // 0 to 0.5: blue to purple to red
        // 0.5 to 1: red to yellow to white

        if log_mag < -2.0 {
            // Very small magnitudes: black to dark blue
            let t = (log_mag + 10.0) / 8.0; // 0 to 1 as we go from -10 to -2
            let intensity = (t * 128.0) as u8;
            Color::RGB(0, 0, intensity)
        } else if log_mag < 0.0 {
            // Small magnitudes: dark blue to bright blue/purple
            let t = (log_mag + 2.0) / 2.0; // 0 to 1 as we go from -2 to 0
            let blue = (128.0 + t * 127.0) as u8;
            let red = (t * 128.0) as u8;
            Color::RGB(red, 0, blue)
        } else if log_mag < 0.5 {
            // Medium magnitudes: purple to red to orange
            let t = log_mag / 0.5; // 0 to 1
            let red = (128.0 + t * 127.0) as u8;
            let green = (t * 128.0) as u8;
            let blue = ((1.0 - t) * 255.0) as u8;
            Color::RGB(red, green, blue)
        } else {
            // Large magnitudes: orange to yellow to white
            let t = (log_mag - 0.5) / 0.5; // 0 to 1
            let green = (128.0 + t * 127.0) as u8;
            Color::RGB(255, green, (t * 255.0) as u8)
        }
    } else {
        // Linear scale - original color mapping
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
}

fn complex_pow(base: &Complex, exponent: &Complex) -> Complex {
    let ln_base = base.clone().ln();
    let product = exponent * ln_base;
    product.exp()
}

fn riemann_zeta(s: &Complex) -> Complex {
    let precision = s.prec().0;
    let one = Complex::with_val(precision, (1.0, 0.0));

    // Check for pole at s=1
    let s_minus_one = Complex::with_val(precision, s - &one);
    if s_minus_one.real().clone().abs() < 1e-10 && s_minus_one.imag().clone().abs() < 1e-10 {
        return Complex::with_val(precision, (Float::with_val(precision, 1e100), 0.0));
    }

    // For Re(s) < 0, return zero
    if s.real().to_f64() < 0.0 {
        return Complex::with_val(precision, (0.0, 0.0));
    }

    // For Re(s) >= 0, use the eta function method
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

    let max_terms = ((precision as f64 / 2.0).ceil() as usize)
        .max(100)
        .min(50000);

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
