import random


def generate_config():
    algorithms = ["CONSTRUCTIVE_DYNAMIC", "CONSTRUCTIVE_STATIC"]
    beta = ["LOW", "MEDIUM", "HIGH"]
    instance = ['p6.2.a.txt', 'p6.2.b.txt', 'p6.2.c.txt', 'p6.2.d.txt',
                'p6.2.e.txt', 'p6.2.f.txt', 'p6.2.g.txt', 'p6.2.h.txt',
                'p6.2.i.txt', 'p6.2.j.txt', 'p6.2.k.txt', 'p6.2.l.txt',
                'p6.2.m.txt', 'p6.2.n.txt', 'p6.3.a.txt', 'p6.3.b.txt',
                'p6.3.c.txt', 'p6.3.d.txt', 'p6.3.e.txt', 'p6.3.f.txt',
                'p6.3.g.txt', 'p6.3.h.txt', 'p6.3.i.txt', 'p6.3.j.txt',
                'p6.3.k.txt', 'p6.3.l.txt', 'p6.3.m.txt', 'p6.3.n.txt',
                'p6.4.a.txt', 'p6.4.b.txt', 'p6.4.c.txt', 'p6.4.d.txt',
                'p6.4.e.txt', 'p6.4.f.txt', 'p6.4.g.txt', 'p6.4.h.txt',
                'p6.4.i.txt', 'p6.4.j.txt', 'p6.4.k.txt', 'p6.4.l.txt',
                'p6.4.m.txt', 'p6.4.n.txt']

    instance += ['p5.2.a.txt', 'p5.2.b.txt', 'p5.2.c.txt', 'p5.2.d.txt',
                 'p5.2.e.txt', 'p5.2.f.txt', 'p5.2.g.txt', 'p5.2.h.txt',
                 'p5.2.i.txt', 'p5.2.j.txt', 'p5.2.k.txt', 'p5.2.l.txt',
                 'p5.2.m.txt', 'p5.2.n.txt', 'p5.2.o.txt', 'p5.2.p.txt',
                 'p5.2.q.txt', 'p5.2.r.txt', 'p5.2.s.txt', 'p5.2.t.txt',
                 'p5.2.u.txt', 'p5.2.v.txt', 'p5.2.w.txt', 'p5.2.x.txt',
                 'p5.2.y.txt', 'p5.2.z.txt', 'p5.3.a.txt', 'p5.3.b.txt',
                 'p5.3.c.txt', 'p5.3.d.txt', 'p5.3.e.txt', 'p5.3.f.txt',
                 'p5.3.g.txt', 'p5.3.h.txt', 'p5.3.i.txt', 'p5.3.j.txt',
                 'p5.3.k.txt', 'p5.3.l.txt', 'p5.3.m.txt', 'p5.3.n.txt',
                 'p5.3.o.txt', 'p5.3.p.txt', 'p5.3.q.txt', 'p5.3.r.txt',
                 'p5.3.s.txt', 'p5.3.t.txt', 'p5.3.u.txt', 'p5.3.v.txt',
                 'p5.3.w.txt', 'p5.3.x.txt', 'p5.3.y.txt', 'p5.3.z.txt',
                 'p5.4.a.txt', 'p5.4.b.txt', 'p5.4.c.txt', 'p5.4.d.txt',
                 'p5.4.e.txt', 'p5.4.f.txt', 'p5.4.g.txt', 'p5.4.h.txt',
                 'p5.4.i.txt', 'p5.4.j.txt', 'p5.4.k.txt', 'p5.4.l.txt',
                 'p5.4.m.txt', 'p5.4.n.txt', 'p5.4.o.txt', 'p5.4.p.txt',
                 'p5.4.q.txt', 'p5.4.r.txt', 'p5.4.s.txt', 'p5.4.t.txt',
                 'p5.4.u.txt', 'p5.4.v.txt', 'p5.4.w.txt', 'p5.4.x.txt',
                 'p5.4.y.txt', 'p5.4.z.txt']

    instance += ['p4.2.a.txt', 'p4.2.b.txt', 'p4.2.c.txt', 'p4.2.d.txt',
                 'p4.2.e.txt', 'p4.2.f.txt', 'p4.2.g.txt', 'p4.2.h.txt',
                 'p4.2.i.txt', 'p4.2.j.txt', 'p4.2.k.txt', 'p4.2.l.txt',
                 'p4.2.m.txt', 'p4.2.n.txt', 'p4.2.o.txt', 'p4.2.p.txt',
                 'p4.2.q.txt', 'p4.2.r.txt', 'p4.2.s.txt', 'p4.2.t.txt',
                 'p4.3.a.txt', 'p4.3.b.txt', 'p4.3.c.txt', 'p4.3.d.txt',
                 'p4.3.e.txt', 'p4.3.f.txt', 'p4.3.g.txt', 'p4.3.h.txt',
                 'p4.3.i.txt', 'p4.3.j.txt', 'p4.3.k.txt', 'p4.3.l.txt',
                 'p4.3.m.txt', 'p4.3.n.txt', 'p4.3.o.txt', 'p4.3.p.txt',
                 'p4.3.q.txt', 'p4.3.r.txt', 'p4.3.s.txt', 'p4.3.t.txt',
                 'p4.4.a.txt', 'p4.4.b.txt', 'p4.4.c.txt', 'p4.4.d.txt',
                 'p4.4.e.txt', 'p4.4.f.txt', 'p4.4.g.txt', 'p4.4.h.txt',
                 'p4.4.i.txt', 'p4.4.j.txt', 'p4.4.k.txt', 'p4.4.l.txt',
                 'p4.4.m.txt', 'p4.4.n.txt', 'p4.4.o.txt', 'p4.4.p.txt',
                 'p4.4.q.txt', 'p4.4.r.txt', 'p4.4.s.txt', 'p4.4.t.txt']

    instance += ['p7.2.a.txt', 'p7.2.b.txt', 'p7.2.c.txt', 'p7.2.d.txt',
                 'p7.2.e.txt', 'p7.2.f.txt', 'p7.2.g.txt', 'p7.2.h.txt',
                 'p7.2.i.txt', 'p7.2.j.txt', 'p7.2.k.txt', 'p7.2.l.txt',
                 'p7.2.m.txt', 'p7.2.n.txt', 'p7.2.o.txt', 'p7.2.p.txt',
                 'p7.2.q.txt', 'p7.2.r.txt', 'p7.2.s.txt', 'p7.2.t.txt',
                 'p7.3.a.txt', 'p7.3.b.txt', 'p7.3.c.txt', 'p7.3.d.txt',
                 'p7.3.e.txt', 'p7.3.f.txt', 'p7.3.g.txt', 'p7.3.h.txt',
                 'p7.3.i.txt', 'p7.3.j.txt', 'p7.3.k.txt', 'p7.3.l.txt',
                 'p7.3.m.txt', 'p7.3.n.txt', 'p7.3.o.txt', 'p7.3.p.txt',
                 'p7.3.q.txt', 'p7.3.r.txt', 'p7.3.s.txt', 'p7.3.t.txt',
                 'p7.4.a.txt', 'p7.4.b.txt', 'p7.4.c.txt', 'p7.4.d.txt',
                 'p7.4.e.txt', 'p7.4.f.txt', 'p7.4.g.txt', 'p7.4.h.txt',
                 'p7.4.i.txt', 'p7.4.j.txt', 'p7.4.k.txt', 'p7.4.l.txt',
                 'p7.4.m.txt', 'p7.4.n.txt', 'p7.4.o.txt', 'p7.4.p.txt',
                 'p7.4.q.txt', 'p7.4.r.txt', 'p7.4.s.txt', 'p7.4.t.txt']
    seeds = [random.randint(100000, 999999) for _ in range(3)]
    with open('run.txt', 'w') as file:
        for i in instance:
            for a in algorithms:
                for b in ["LOW", "MEDIUM", "HIGH"]:
                    for seed in seeds:  # Tres semillas aleatorias
                        if i[1] == "7":
                            max_iter_dynamic = 1000
                            max_iter_random = 1000
                        else:
                            max_iter_dynamic = 100
                            max_iter_random = 100

                        max_time = 1
                        algorithm = a
                        selected_random_node = "GRASP"
                        alpha = 0.9
                        percentage = 0.9
                        n_type_nodes = 5
                        beta_bias = 0.7
                        standardize = True

                        # Escribiendo cada configuración en el archivo
                        file.write(
                            f"{i};{seed};{max_iter_dynamic};{max_time};{algorithm};{selected_random_node};{b};{max_iter_random};{alpha};{percentage};{n_type_nodes};{beta_bias};{standardize}\n")

generate_config()