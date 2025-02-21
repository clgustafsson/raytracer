use ::image as im;
use cgmath::{dot, ElementWise, InnerSpace, Vector3};
use rand::Rng;

const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 450;

const VFOV: f64 = 20.;
const LOOKFROM: Vector3<f64> = Vector3 {
    x: 13.,
    y: 2.,
    z: 3.,
};
const LOOKAT: Vector3<f64> = Vector3 {
    x: 0.,
    y: 0.,
    z: 0.,
};
const VUP: Vector3<f64> = Vector3 {
    x: 0.,
    y: 1.,
    z: 0.,
};
const DEFOCUS_ANGLE: f64 = 0.;
const FOCUS_DIST: f64 = 10.;

const BACKGROUND: bool = false;

const WINDOW_FWIDTH: f64 = WINDOW_WIDTH as f64;
const WINDOW_FHEIGTH: f64 = WINDOW_HEIGHT as f64;

const PI: f64 = 3.1415926535897932385;

fn degree_to_radians(degrees: f64) -> f64 {
    degrees * PI / 180.
}

fn random_0_1() -> f64 {
    return rand::thread_rng().gen_range(0.0..1.0);
}

fn random_in_interval(i: Interval) -> f64 {
    return rand::thread_rng().gen_range(i.min..i.max);
}

fn random_vec() -> Vector3<f64> {
    Vector3::from([random_0_1(), random_0_1(), random_0_1()])
}

fn random_vec_from_interval(i: Interval) -> Vector3<f64> {
    Vector3::from([
        random_in_interval(i),
        random_in_interval(i),
        random_in_interval(i),
    ])
}

fn random_vec_in_unit_sphere() -> Vector3<f64> {
    loop {
        let p = random_vec_from_interval(Interval::from(-1., 1.));
        if p.magnitude2() < 1. {
            return p;
        }
    }
}

fn random_unit_vec() -> Vector3<f64> {
    random_vec_in_unit_sphere().normalize()
}

fn random_on_hemisphere(normal: Vector3<f64>) -> Vector3<f64> {
    let on_unit_sphere = random_unit_vec();
    if dot(on_unit_sphere, normal) > 0. {
        on_unit_sphere
    } else {
        -on_unit_sphere
    }
}

fn random_in_unit_disk() -> Vector3<f64> {
    loop {
        let p = Vector3::from([
            random_in_interval(Interval::from(-1., 1.)),
            random_in_interval(Interval::from(-1., 1.)),
            0.,
        ]);
        if p.magnitude2() < 1. {
            return p;
        }
    }
}

fn linear_to_gamma(linear_component: f64) -> f64 {
    linear_component.sqrt()
}

fn reflect(v: Vector3<f64>, normal: Vector3<f64>) -> Vector3<f64> {
    v - 2. * dot(v, normal) * normal
}

fn refract(uv: Vector3<f64>, normal: Vector3<f64>, etai_over_etat: f64) -> Vector3<f64> {
    let cos_theta = f64::min(dot(-uv, normal), 1.);
    let r_out_perp = etai_over_etat * (uv + cos_theta * normal);
    let r_out_parallel = -(1.0 - r_out_perp.magnitude2()).abs().sqrt() * normal;
    r_out_perp + r_out_parallel
}

fn reflectance(cosine: f64, relection_i: f64) -> f64 {
    //Schlicks approximation
    let mut r0 = (1. - relection_i) / (1. + relection_i);
    r0 = r0 * r0;
    r0 + (1. - r0) * (1. - cosine) * (1. - cosine) * (1. - cosine) * (1. - cosine) * (1. - cosine)
}

#[derive(Clone, Copy)]
struct Interval {
    min: f64,
    max: f64,
}

impl Interval {
    fn new() -> Interval {
        Interval {
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }
    fn from(min: f64, max: f64) -> Interval {
        Interval { min: min, max: max }
    }

    fn contains(&self, n: f64) -> bool {
        return self.min <= n && n <= self.max;
    }

    fn surrounds(&self, n: f64) -> bool {
        return self.min < n && n < self.max;
    }

    fn clamp(&self, n: f64) -> f64 {
        if n < self.min {
            return self.min;
        }
        if n > self.max {
            return self.max;
        }
        n
    }
}

const EMPTY: Interval = Interval {
    min: f64::INFINITY,
    max: f64::NEG_INFINITY,
};

const UNIVERSE: Interval = Interval {
    min: f64::NEG_INFINITY,
    max: f64::INFINITY,
};

fn rgb_to_color(mut r: f64, mut g: f64, mut b: f64, samples: i32) -> [u8; 4] {
    let scale = 1. / samples as f64;
    r *= scale;
    g *= scale;
    b *= scale;

    r = linear_to_gamma(r);
    g = linear_to_gamma(g);
    b = linear_to_gamma(b);

    let intensity = Interval::from(0., 0.999);
    [
        (255.99 * intensity.clamp(r)) as u8,
        (255.99 * intensity.clamp(g)) as u8,
        (255.99 * intensity.clamp(b)) as u8,
        255,
    ]
}

#[derive(Clone, Copy)]
enum MaterialType<Color, Fuzz, RefractionIndex> {
    Lambertian(Color),
    Metal(Color, Fuzz),
    Dielectric(RefractionIndex),
    DiffuseLight,
    None,
}

#[derive(Clone, Copy)]
struct Material {
    material_type: MaterialType<Vector3<f64>, f64, f64>,
}
impl Material {
    fn scatter(
        &self,
        ray: Ray,
        rec: HitRecord,
        attenuation: &mut Vector3<f64>,
        scattered: &mut Ray,
    ) -> bool {
        match self.material_type {
            MaterialType::Lambertian(color) => {
                let mut scatter_direction = rec.normal + random_unit_vec();
                if near_zero(scatter_direction) {
                    scatter_direction = rec.normal;
                }
                *scattered = Ray::new(rec.position, scatter_direction);
                *attenuation = color;
                true
            }
            MaterialType::Metal(color, fuzz) => {
                let reflected = reflect(ray.direction.normalize(), rec.normal);
                *scattered = Ray::new(rec.position, reflected + fuzz * random_unit_vec());
                *attenuation = color;
                dot(scattered.direction, rec.normal) > 0.
            }
            MaterialType::Dielectric(ir) => {
                *attenuation = Vector3::from([1., 1., 1.]);
                let refraction_ratio = if rec.front_face { 1. / ir } else { ir };
                let unit_direction = ray.direction.normalize();

                let cos_theta = f64::min(dot(-unit_direction, rec.normal), 1.);
                let sin_theta = (1. - cos_theta * cos_theta).sqrt();

                let cannot_refract = refraction_ratio * sin_theta > 1.;
                let direction;

                if cannot_refract || reflectance(cos_theta, refraction_ratio) > random_0_1() {
                    direction = reflect(unit_direction, rec.normal)
                } else {
                    direction = refract(unit_direction, rec.normal, refraction_ratio)
                }

                *scattered = Ray::new(rec.position, direction);
                true
            }
            MaterialType::DiffuseLight => false,
            _ => true,
        }
    }

    fn emmited(&self) -> Vector3<f64> {
        match self.material_type {
            MaterialType::DiffuseLight => Vector3::from([1., 1., 1.]),
            _ => Vector3::from([0., 0., 0.]),
        }
    }
}

fn near_zero(v: Vector3<f64>) -> bool {
    let limit = 1e-8;
    v.x < limit && v.y < limit && v.z < limit
}

#[derive(Clone, Copy)]
struct HitRecord {
    position: Vector3<f64>,
    normal: Vector3<f64>,
    material: Material,
    t: f64,
    front_face: bool,
}

impl HitRecord {
    fn new() -> HitRecord {
        HitRecord {
            position: Vector3::from([0., 0., 0.]),
            normal: Vector3::from([0., 0., 0.]),
            material: Material {
                material_type: MaterialType::None,
            },
            t: 0.,
            front_face: true,
        }
    }

    fn set_face_normal(&mut self, r: Ray, outward_normal: Vector3<f64>) {
        self.front_face = dot(r.direction, outward_normal) < 0.;
        self.normal = if self.front_face {
            outward_normal
        } else {
            -outward_normal
        };
    }
}
#[derive(Clone)]
struct Sphere {
    position: Vector3<f64>,
    radius: f64,
    material: Material,
}
#[derive(Clone)]
struct World {
    spheres: Vec<Sphere>,
}

impl World {
    fn new() -> World {
        World { spheres: vec![] }
    }

    fn hit_spheres(&self, r: Ray, ray_t: Interval, rec: &mut HitRecord) -> bool {
        let mut temp_rec = HitRecord::new();
        let mut hit_anything: bool = false;
        let mut closest_so_far = ray_t.max;

        for sphere in &self.spheres {
            if r.hit_sphere(
                sphere,
                Interval::from(ray_t.min, closest_so_far),
                &mut temp_rec,
            ) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                *rec = temp_rec;
            }
        }
        hit_anything
    }
}

#[derive(Clone, Copy)]
struct Ray {
    origin: Vector3<f64>,
    direction: Vector3<f64>,
}

impl Ray {
    fn new(origin: Vector3<f64>, direction: Vector3<f64>) -> Ray {
        Ray { origin, direction }
    }
    fn at(&self, t: f64) -> Vector3<f64> {
        self.origin + t * self.direction
    }
    fn hit_sphere(&self, sphere: &Sphere, ray_t: Interval, rec: &mut HitRecord) -> bool {
        let oc = self.origin - sphere.position;
        let a: f64 = self.direction.magnitude2();
        let half_b = dot(oc, self.direction);
        let c = oc.magnitude2() - sphere.radius * sphere.radius;
        let discriminant = half_b * half_b - a * c;
        if discriminant < 0. {
            return false;
        }
        let sqrtd = discriminant.sqrt();
        let mut root = (-half_b - sqrtd) / a;
        if !ray_t.surrounds(root) {
            root = (-half_b + sqrtd) / a;
            if !ray_t.surrounds(root) {
                return false;
            }
        }
        rec.t = root;
        rec.position = self.at(rec.t);
        rec.normal = (rec.position - sphere.position) / sphere.radius;
        let outward_normal = (rec.position - sphere.position) / sphere.radius;
        rec.set_face_normal(*self, outward_normal);
        rec.material = sphere.material;

        true
    }
}
fn ray_color(r: Ray, depth: i32, world: World) -> Vector3<f64> {
    let mut rec = HitRecord::new();
    if depth <= 0 {
        return Vector3::from([0., 0., 0.]);
    }

    if world.hit_spheres(r, Interval::from(0.001, f64::INFINITY), &mut rec) {
        let mut scattered = r;
        let mut attenuation = Vector3::from([0., 0., 0.]);
        let color_from_emission = rec.material.emmited();
        if !rec
            .material
            .scatter(r, rec, &mut attenuation, &mut scattered)
        {
            return color_from_emission;
        }
        let color_from_scatter =
            attenuation.mul_element_wise(ray_color(scattered, depth - 1, world));
        return color_from_emission + color_from_scatter;
    } else {
        if BACKGROUND {
            let unit_direction = r.direction.normalize();
            let a = 0.5 * (unit_direction.y + 1.);
            (1. - a) * Vector3::from([1., 1., 1.]) + a * Vector3::from([0.5, 0.7, 1.])
        } else {
            Vector3::from([0., 0., 0.])
        }
    }
}

fn render(
    canvas: &mut im::ImageBuffer<im::Rgba<u8>, Vec<u8>>,
    world: World,
    samples_per_pixel: i32,
    max_depth: i32,
) {
    let camera_position = LOOKFROM;

    let vfov = VFOV;

    let theta = degree_to_radians(vfov);
    let h = (theta / 2.).tan();
    let viewport_height: f64 = 2. * h * FOCUS_DIST;
    let viewport_width = viewport_height * WINDOW_FWIDTH / WINDOW_FHEIGTH;

    let w = (LOOKFROM - LOOKAT).normalize();
    let u = VUP.cross(w).normalize();
    let v = w.cross(u);

    let viewport_u: Vector3<f64> = viewport_width * u;
    let viewport_v: Vector3<f64> = viewport_height * -v;

    let pixel_delta_u = viewport_u / WINDOW_FWIDTH;
    let pixel_delta_v = viewport_v / WINDOW_FHEIGTH;

    let viewport_top_left = camera_position - (FOCUS_DIST * w) - viewport_u / 2. - viewport_v / 2.;
    let pixel00_loc = viewport_top_left + (pixel_delta_u + pixel_delta_v) / 2.;

    let defocus_radius = FOCUS_DIST * (degree_to_radians(DEFOCUS_ANGLE / 2.).tan());
    let defocus_disk_u = u * defocus_radius;
    let defocus_disk_v = v * defocus_radius;

    for x in 0..WINDOW_WIDTH {
        for y in 0..WINDOW_HEIGHT {
            let mut pixel_color = Vector3::from([0., 0., 0.]);
            for _ in 0..samples_per_pixel {
                let r = get_ray(
                    x,
                    y,
                    camera_position,
                    pixel00_loc,
                    pixel_delta_u,
                    pixel_delta_v,
                    defocus_disk_u,
                    defocus_disk_v,
                );

                //dbg!(pixel_color, sample_color);
                pixel_color += ray_color(r, max_depth, world.clone());
            }
            let color = rgb_to_color(
                pixel_color.x,
                pixel_color.y,
                pixel_color.z,
                samples_per_pixel,
            );
            canvas.put_pixel(x, y, im::Rgba(color));
        }
    }
}

fn get_ray(
    x: u32,
    y: u32,
    camera_position: Vector3<f64>,
    pixel00_loc: Vector3<f64>,
    pixel_delta_u: Vector3<f64>,
    pixel_delta_v: Vector3<f64>,
    defocus_disk_u: Vector3<f64>,
    defocus_disk_v: Vector3<f64>,
) -> Ray {
    let pixel_center = pixel00_loc + pixel_delta_u * x as f64 + pixel_delta_v * y as f64;
    let pixel_sample = pixel_center + pixel_sample_square(pixel_delta_u, pixel_delta_v);

    let ray_origin = if DEFOCUS_ANGLE <= 0. {
        camera_position
    } else {
        defocus_disk_sample(camera_position, defocus_disk_u, defocus_disk_v)
    };
    let ray_direction = pixel_sample - ray_origin;

    Ray::new(camera_position, ray_direction)
}

fn defocus_disk_sample(
    position: Vector3<f64>,
    defocus_disk_u: Vector3<f64>,
    defocus_disk_v: Vector3<f64>,
) -> Vector3<f64> {
    let p = random_in_unit_disk();
    position + (p.x * defocus_disk_u) + (p.y * defocus_disk_v)
}

fn pixel_sample_square(pixel_delta_u: Vector3<f64>, pixel_delta_v: Vector3<f64>) -> Vector3<f64> {
    let px = -0.5 + random_0_1();
    let py = -0.5 + random_0_1();
    (px * pixel_delta_u) + (py * pixel_delta_v)
}

fn main() {
    let mut canvas = im::ImageBuffer::new(WINDOW_WIDTH, WINDOW_HEIGHT);

    let mut world = World::new();

    let material_ground = MaterialType::Lambertian(Vector3::from([0.5, 0.5, 0.5]));
    world.spheres.push(Sphere {
        position: Vector3::from([0., -1000., 0.]),
        radius: 1000.,
        material: Material {
            material_type: material_ground,
        },
    });

    for a in -11..11 {
        for b in -11..11 {
            let choose_material = random_0_1();
            let center = Vector3::from([
                a as f64 + 0.9 * random_0_1(),
                0.2,
                b as f64 + 0.9 * random_0_1(),
            ]);

            if (center - Vector3::from([4., 0.2, 0.])).magnitude() > 0.9 {
                if choose_material < 0.4 {
                    let light = MaterialType::DiffuseLight;
                    world.spheres.push(Sphere {
                        position: center,
                        radius: 0.2,
                        material: Material {
                            material_type: light,
                        },
                    });
                }
                if choose_material < 0.8 {
                    let color = Vector3::from([
                        random_in_interval(Interval::from(0., 1.)),
                        random_in_interval(Interval::from(0., 1.)),
                        random_in_interval(Interval::from(0., 1.)),
                    ]);
                    let sphere_material = MaterialType::Lambertian(color);
                    world.spheres.push(Sphere {
                        position: center,
                        radius: 0.2,
                        material: Material {
                            material_type: sphere_material,
                        },
                    });
                } else if choose_material < 0.95 {
                    let color = Vector3::from([
                        random_in_interval(Interval::from(0.5, 1.)),
                        random_in_interval(Interval::from(0., 1.)),
                        random_in_interval(Interval::from(0., 1.)),
                    ]);
                    let fuzz = random_in_interval(Interval::from(0., 0.5));
                    let sphere_material = MaterialType::Metal(color, fuzz);
                    world.spheres.push(Sphere {
                        position: center,
                        radius: 0.2,
                        material: Material {
                            material_type: sphere_material,
                        },
                    });
                } else {
                    let sphere_material = MaterialType::Dielectric(1.5);
                    world.spheres.push(Sphere {
                        position: center,
                        radius: 0.2,
                        material: Material {
                            material_type: sphere_material,
                        },
                    });
                }
            }
        }
    }

    let material1 = MaterialType::Dielectric(1.5);
    world.spheres.push(Sphere {
        position: Vector3::from([0., 1., 0.]),
        radius: 1.,
        material: Material {
            material_type: material1,
        },
    });

    let material2 = MaterialType::Lambertian(Vector3::from([0.4, 0.2, 0.1]));
    world.spheres.push(Sphere {
        position: Vector3::from([-4., 1., 0.]),
        radius: 1.,
        material: Material {
            material_type: material2,
        },
    });

    let material3 = MaterialType::Metal(Vector3::from([0.7, 0.6, 0.5]), 0.0);
    world.spheres.push(Sphere {
        position: Vector3::from([4., 1., 0.]),
        radius: 1.,
        material: Material {
            material_type: material3,
        },
    });

    let light = MaterialType::DiffuseLight;
    world.spheres.push(Sphere {
        position: Vector3::from([0., 10., 0.]),
        radius: 5.,
        material: Material {
            material_type: light,
        },
    });

    render(&mut canvas, world.clone(), 100, 25);
    canvas.save("render.png").unwrap();
}
