For information about Liquid Earth, check the commit descriptions in the non-cartogram-hybrid branch.


To create a plane cartogram, run:
    with open("oct32-tris-to-cut-list-excl.pickle", "rb") as f:
        tris_to_cut_list = pickle.load(f)
    mesh, verts_proj = subdivide_octahedron_interrupted(32, 0, False)
    mesh_proj = Mesh(verts_proj, mesh.tris)
    for tris_to_cut in tris_to_cut_list:
        mesh = subdivide_mesh_selected(mesh, tris_to_cut)
        mesh_proj = subdivide_mesh_selected(mesh_proj, tris_to_cut)
    portions = np.load("oct32-subdiv-shifted-50m-portions.npy")
    cart = cartogram(mesh, portions, POPULATION_ARRAY, initial_verts=mesh_proj.verts)
	
	"""Stuff for drawing the cartogram below"""
	ang = np.deg2rad(11)
    mat = np.array([[np.cos(ang), np.sin(ang), 0], [-np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    poly_list, is_closed_list = poly_list_from_borders(WORLD_BORDERS_DATA_FLAT)
    
    poly_list = [matrix_times_array_of_vectors(mat, poly) for poly in poly_list]
    poly_list, is_closed_list = interrupt_polygon_list_antimeridian(poly_list,
                                                                is_closed_list,
                                                                shift_degrees=0)
    lon_line_ll_0 = np.column_stack([10000 * [-np.pi + 1e-6],
                                np.linspace(-np.pi/2, np.pi/2, 10000, True)])
    lon_line_ll_1 = np.column_stack([10000 * [np.pi - 1e-6],
                                np.linspace(-np.pi/2, np.pi/2, 10000, True)])
    poly_list.append(lonlat_to_cartes(lon_line_ll_0.T).T)
    poly_list.append(lonlat_to_cartes(lon_line_ll_1.T).T)
    is_closed_list += [False, False]
    poly_list = polys_old_mesh_to_new(poly_list, mesh, cart)
    set_up_plot()
    plot_polygons(poly_list, is_closed_list)
    #plot_mesh(cart)
	
    xlim = 4
    ylim = 2
    _ = plt.gca().set_xlim(-xlim, xlim)
    _ = plt.gca().set_ylim(-ylim, ylim)
	
To create a sphere cartogram, run:
	proj_derivs = mollweide_derivs # change this to whatever target projection is desired
    with open("oct32-tris-to-cut-list-excl.pickle", "rb") as f:
        tris_to_cut_list = pickle.load(f)
    #with open("oct32-portions-list-50m.pickle", "rb") as f:
    #    portions_list = pickle.load(f)
    mesh = subdivide_mesh_sphere(OCTAHEDRON, 32)
    for tris_to_cut in tris_to_cut_list:
        mesh = subdivide_mesh_selected(mesh, tris_to_cut)
    #portions = portions_list[-1]
    portions = np.load("oct32-subdiv-shifted-50m-portions.npy")
    cart = cartogram(mesh, portions, POPULATION_ARRAY, sphere_first=True)
	
	"""Stuff for drawing the cartogram below"""
	ang = np.deg2rad(11)
    mat = np.array([[np.cos(ang), np.sin(ang), 0], [-np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    def proj(lonlat):
        return pseudocylindrical(proj_derivs, lonlat)
    poly_list, is_closed_list = poly_list_from_borders(WORLD_BORDERS_DATA_FLAT)
    
    poly_list = [matrix_times_array_of_vectors(mat, poly) for poly in poly_list]
    poly_list_cart = polys_old_mesh_to_new(poly_list, mesh, cart)
    poly_list, is_closed_list = interrupt_polygon_list_antimeridian(poly_list_cart,
                                                                is_closed_list,
                                                                shift_degrees=0)
    poly_list = [proj(cartes_to_lonlat(poly.T)).T for poly in poly_list]
    set_up_plot()
    plot_polygons(poly_list, is_closed_list)

    edge = np.linspace(-np.pi/2, np.pi/2, 1000)
    edge1 = np.array([np.pi + 0*edge, edge])
    edge2 = np.array([-np.pi + 0*edge, edge]).T[::-1].T
    edge = np.concatenate((edge1, edge2), axis=1)
    plot_polygons([proj(edge).T], [True])
    xlim = proj(np.array([np.pi, 0]))[0] + 0.01
    ylim = proj(np.array([0, np.pi/2]))[1] + 0.01
    _ = plt.gca().set_xlim(-xlim, xlim)
    _ = plt.gca().set_ylim(-ylim, ylim)

To create a hybrid cartogram, run:
	proj_derivs = mollweide_derivs # change this to whatever target projection is desired
    with open("oct32-tris-to-cut-list-excl.pickle", "rb") as f:
        tris_to_cut_list = pickle.load(f)
    #with open("oct32-portions-list-50m.pickle", "rb") as f:
    #    portions_list = pickle.load(f)
    mesh = subdivide_mesh_sphere(OCTAHEDRON, 32)
    for tris_to_cut in tris_to_cut_list:
        mesh = subdivide_mesh_selected(mesh, tris_to_cut)
    #portions = portions_list[-1]
    portions = np.load("oct32-subdiv-shifted-50m-portions.npy")
    cart = cartogram(mesh, portions, POPULATION_ARRAY, sphere_first=True, hybrid=True, fix_antimer=True, proj_derivs=proj_derivs)
	
	"""Stuff for drawing the cartogram below"""
	ang = np.deg2rad(11)
    mat = np.array([[np.cos(ang), np.sin(ang), 0], [-np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    def proj(lonlat):
        return pseudocylindrical(proj_derivs, lonlat)
    poly_list, is_closed_list = poly_list_from_borders(WORLD_BORDERS_DATA_FLAT)
    
    
    poly_list = [matrix_times_array_of_vectors(mat, poly) for poly in poly_list]
    poly_list_cart = polys_old_mesh_to_new(poly_list, mesh, cart)
    poly_list, is_closed_list = interrupt_polygon_list_antimeridian(poly_list_cart,
                                                                is_closed_list,
                                                                shift_degrees=0)
    poly_list = [proj(cartes_to_lonlat(poly.T)).T for poly in poly_list]
    set_up_plot()
    plot_polygons(poly_list, is_closed_list)

    edge = np.linspace(-np.pi/2, np.pi/2, 1000)
    edge1 = np.array([np.pi + 0*edge, edge])
    edge2 = np.array([-np.pi + 0*edge, edge]).T[::-1].T
    edge = np.concatenate((edge1, edge2), axis=1)
    plot_polygons([proj(edge).T], [True])
    xlim = proj(np.array([np.pi, 0]))[0] + 0.01
    ylim = proj(np.array([0, np.pi/2]))[1] + 0.01
    _ = plt.gca().set_xlim(-xlim, xlim)
    _ = plt.gca().set_ylim(-ylim, ylim)