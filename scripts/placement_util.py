from threading import local
import numpy as np
import open3d as o3d
import copy
import math

class DirectLine():
    def __init__(self):
        self.translation = None
        self.rotation = None
        self.start = None
        self.end = None
        self.unit = None
        

class BoxContainer():
    def __init__(self):
        self.dx = 0
        self.dy = 0
        self.dz = 0
        self.x = 0
        self.y = 0
        self.z = 0
        self.rotation = np.eye(3)
        self.pcd = None
        self.obbox = None
        self.direction_x = None
        self.direction_y = None
        self.direction_z = None
        self.items = []
        self.prompt_box = None
        self.volume = 0
        self.acc_space = 0

    def set_pcd(self, pcd):
        self.pcd = pcd

    def add_item(self, it):
        self.items.append(it)

    def est_acc_space(self):
        self.acc_space = 0
    
    def set_obbox(self, extent, translation, rotation):
        self.dx = extent[0]
        self.dy = extent[1]
        self.dz = extent[2]
        self.x = translation[0]
        self.y = translation[1]
        self.z = translation[2]
        self.rotaiton = rotation 
        self.obbox = o3d.geometry.OrientedBoundingBox(translation, rotation, extent)
        self.volume = self.obbox.extent[0] * self.obbox.extent[1] * self.obbox.extent[2]
        
        self.direction_y = DirectLine()
        center = np.array([[self.obbox.center[0]],[self.obbox.center[1]],[self.obbox.center[2]]])
        self.direction_y.start = np.array([[0], [-self.obbox.extent[1]/2], [0]])
        self.direction_y.start = self.obbox.R @ self.direction_y.start + center
        self.direction_y.translation = self.direction_y.start.T[0]
        self.direction_y.end = np.array([[0], [self.obbox.extent[1]/2], [0]])
        self.direction_y.end = self.obbox.R @ self.direction_y.end + center
        self.direction_y.unit = (self.direction_y.end - self.direction_y.start).T[0]
        self.direction_y.unit = self.direction_y.unit / np.linalg.norm(self.direction_y.unit)
        local = np.array([[1,0,0],[0,0,1],[0,-1,0]])
        self.direction_y.rotation = self.obbox.R @ local

        self.direction_x = DirectLine()
        center = np.array([[self.obbox.center[0]],[self.obbox.center[1]],[self.obbox.center[2]]])
        self.direction_x.start = np.array([[-self.obbox.extent[0]/2], [0], [0]])
        self.direction_x.start = self.obbox.R @ self.direction_x.start + center
        self.direction_x.translation = self.direction_x.start.T[0]
        self.direction_x.end = np.array([[self.obbox.extent[0]/2], [0], [0]])
        self.direction_x.end = self.obbox.R @ self.direction_x.end + center
        self.direction_x.unit = (self.direction_x.end - self.direction_x.start).T[0]
        self.direction_x.unit = self.direction_x.unit / np.linalg.norm(self.direction_x.unit)
        local = np.array([[0,0,1],[0,1,0],[-1,0,0]])
        self.direction_x.rotation = self.obbox.R @ local
        
        self.direction_z = DirectLine()
        center = np.array([[self.obbox.center[0]],[self.obbox.center[1]],[self.obbox.center[2]]])
        self.direction_z.start = np.array([[0], [0], [self.obbox.extent[2]/2]])
        self.direction_z.start = self.obbox.R @ self.direction_z.start + center
        self.direction_z.translation = self.direction_z.start.T[0]
        self.direction_z.end = np.array([[0], [0], [-self.obbox.extent[2]/2]])
        self.direction_z.end = self.obbox.R @ self.direction_z.end + center
        self.direction_z.unit = (self.direction_z.end - self.direction_z.start).T[0]
        self.direction_z.unit = self.direction_z.unit / np.linalg.norm(self.direction_z.unit)
        local = np.array([[-1,0,0],[0,1,0],[0,0,-1]])
        self.direction_z.rotation = self.obbox.R @ local

    def gt_empty(self):
        rem_space = self.volume
        for pkg in self.items:
            rem_space -= pkg.volume
        return rem_space

    def acc_empty(self, unit = 0.02):
        new_pcd = copy.deepcopy(self.pcd)
        new_pcd.translate(-self.obbox.center)
        inv_rotate = np.linalg.inv(self.obbox.R)
        new_pcd.rotate(inv_rotate, center = [0,0,0])


        extent = [unit, unit, self.obbox.extent[2]]
        rotation = np.array([[1,0,0],[0,1,0],[0,0,1]])
        center = np.array([-self.obbox.extent[0]/2+unit/2,-self.obbox.extent[1]/2+unit/2,0])
        grid = o3d.geometry.OrientedBoundingBox(center, rotation, extent)

        x_count = int(self.obbox.extent[0]/unit)
        y_count = int(self.obbox.extent[1]/unit)
        ceiling = self.obbox.extent[2]/2
        self.acc_space = 0
        for i in range(0, x_count):
            for j in range(0, y_count):
                local_grid = copy.deepcopy(grid)
                local_grid.translate([i * unit, j * unit, 0])
                local_pcd = new_pcd.crop(local_grid)
                local_points = np.asarray(local_pcd.points)
                # if i == 10 and j == 15:
                #     vis = o3d.visualization.Visualizer()
                #     vis.create_window()
                #     vis.add_geometry(new_pcd)
                #     vis.add_geometry(local_grid)
                #     opt = vis.get_render_option()
                #     opt.show_coordinate_frame = True
                #     vis.run()
                #     vis.destroy_window()
                #     print("done")
                if len(local_points) > 5:
                    avg = np.average(local_points, axis=0)
                    self.acc_space += unit * unit * (ceiling-avg[2])

        return self.acc_space

    def estimate_place(self, bbox, z, step = 0.02, side_step = 0.01):
        self.prompt_box = bbox
        rotation = np.array([[1,0,0],[0,0.866,-0.5],[0,0.5,0.866]])
        self.prompt_box.rotate(rotation, center = [0,0,0])
        self.prompt_box.rotate(self.direction_y.rotation, center = [0,0,0])
        self.prompt_box.translate(self.direction_y.translation)
        self.prompt_box.translate([0,0,-z])
        length = self.prompt_box.extent[1] / 4 + self.prompt_box.extent[2]
        self.prompt_box.translate(length*self.direction_y.unit)
        count = int(self.obbox.extent[1] / step)
        result = 0
        good_config = [0,0,0]
        for i in range(0, count):
            flag = False
            for j in range(-5, 6):
                for t in range(-5, 6):
                    local_box = copy.deepcopy(self.prompt_box)
                    local_box.translate(i*step*self.direction_y.unit)
                    local_box.translate(j*side_step*self.direction_x.unit)
                    local_box.translate(t*side_step*self.direction_z.unit)
                    local_pcd = self.pcd.crop(local_box)
                    if (flag == False) and (len(local_pcd.points) < 100):
                        flag = True
                        good_config = [j,i,t]
                        break
                if flag == True:
                    break
            if flag == False:
                break

        self.prompt_box.translate(good_config[1]*step*self.direction_y.unit)
        self.prompt_box.translate(good_config[0]*side_step*self.direction_x.unit)
        self.prompt_box.translate(good_config[2]*side_step*self.direction_z.unit)

        return self.prompt_box

    def crop_box(self):
        self.pcd = self.pcd.crop(self.obbox)

    def viz(self, pcd = True, obbox = True, item = True, direction = True, place_box = True):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        if self.pcd and pcd:
            vis.add_geometry(self.pcd)
        if self.obbox and obbox:
            vis.add_geometry(self.obbox)
        if item and len(self.items):
            for it in self.items:
                if it.pcd:
                    vis.add_geometry(it.pcd)
                if it.mesh_pcd:
                    vis.add_geometry(it.mesh_pcd)
        if self.direction_x is not None and direction:
            red_color = np.array([255, 0, 0])
            length = math.sqrt(np.sum((self.direction_x.start - self.direction_x.end) ** 2))
            arrow = o3d.geometry.TriangleMesh.create_arrow(0.01, 0.015, length, 0.04)
            arrow.paint_uniform_color(red_color / 255)
            arrow.rotate(self.direction_x.rotation, center = [0,0,0])
            arrow.translate(self.direction_x.translation)
            vis.add_geometry(arrow)
        if self.direction_y is not None and direction:
            green_color = np.array([0, 255, 0])
            length = math.sqrt(np.sum((self.direction_y.start - self.direction_y.end) ** 2))
            arrow = o3d.geometry.TriangleMesh.create_arrow(0.01, 0.015, length, 0.04)
            arrow.paint_uniform_color(green_color / 255)
            arrow.rotate(self.direction_y.rotation, center = [0,0,0])
            arrow.translate(self.direction_y.translation)
            vis.add_geometry(arrow)
        if self.direction_z is not None and direction:
            blue_color = np.array([0, 0, 255])
            length = math.sqrt(np.sum((self.direction_z.start - self.direction_z.end) ** 2))
            arrow = o3d.geometry.TriangleMesh.create_arrow(0.01, 0.015, length, 0.04)
            arrow.paint_uniform_color(blue_color / 255)
            arrow.rotate(self.direction_z.rotation, center = [0,0,0])
            arrow.translate(self.direction_z.translation)
            vis.add_geometry(arrow)
        if place_box:
            vis.add_geometry(self.prompt_box)
        opt = vis.get_render_option()
        opt.show_coordinate_frame = True
        vis.run()
        vis.destroy_window()

class package:
    def __init__(self) -> None:
        self.mesh_pcd = None
        self.pcd = None
        self.volume = 0
        self.obbox = None

    def load_from_mesh(self, path): 
        self.mesh_pcd = o3d.io.read_triangle_mesh(path)
        self.mesh_pcd.paint_uniform_color(np.array([0, 255, 0]))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(self.mesh_pcd.vertices))
        self.mesh_pcd = pcd
    
    def get_obbox(self):
        self.obbox = o3d.geometry.OrientedBoundingBox.create_from_points(self.pcd.points)
        self.obbox.color = [1,0,1]


    def load_pcd(self, pcd):
        self.pcd = pcd

    def cal_vlm(self, transform, grid_size = 0.000005):
        local_pcd = copy.deepcopy(self.pcd)
        local_pcd.transform(transform)
        print("getting {} points".format(np.array(local_pcd.points).shape[0]))

        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # vis.add_geometry(local_pcd)
        # x = o3d.geometry.TriangleMesh.create_sphere(radius=0.00002)
        # green_color = np.array([0, 255, 0])
        # x.paint_uniform_color(green_color / 255)
        # vis.add_geometry(x)
        # opt = vis.get_render_option()
        # vis.run()
        # vis.destroy_window()
        
        pt_max = local_pcd.get_max_bound()
        pt_min = local_pcd.get_min_bound()
        print("max {}, min {}".format(pt_max, pt_min))
        print("getting {} points".format(np.array(local_pcd.points).shape[0]))
        x_size = int((pt_max[0] - pt_min[0]) / grid_size) + 1
        y_size = int(pt_max[1] - pt_min[1] / grid_size) + 1
        volume = 0
        grid_count = 0
        grid_avg_z = 0
        point_count = 0
        print("Total grid {} x_size: {} y_size: {}".format(x_size * y_size, x_size, y_size))
        for i in range(0, x_size):
            for j in range(0, y_size):
                min_bound = [pt_min[0]+i*grid_size, pt_min[1]+j*grid_size, -0.001]
                max_bound = [pt_min[0]+(i+1)*grid_size, pt_min[1]+(j+1)*grid_size, pt_max[2]*5]
                bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
                temp_points = np.array(local_pcd.crop(bbox).points)
                point_count += temp_points.shape[0]
                if temp_points.shape[0] > 0:
                    avg = np.average(temp_points, axis = 0)
                    grid_avg_z += avg[2]
                    volume += grid_size * grid_size * avg[2] * 1e9
                    grid_count += 1
        print("Grid in to sys {} avg_size {} point_sum {}".format(grid_count, grid_avg_z/grid_count*1000, point_count))
        return volume

    def viz(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        if self.pcd:
            vis.add_geometry(self.pcd)
        if self.mesh_pcd:
            vis.add_geometry(self.mesh_pcd)
        if self.obbox:
            vis.add_geometry(self.obbox)
        
        r = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        r.translate([1, 0, 0])
        r.paint_uniform_color([1,0,0])
        vis.add_geometry(r)
        g = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        g.translate([0, 1, 0])
        g.paint_uniform_color([0,1,0])
        vis.add_geometry(g)
        b = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        b.translate([0, 0, 1])
        b.paint_uniform_color([0,0,1])
        vis.add_geometry(b)
        y = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        y.translate([0, 0, 0])
        y.paint_uniform_color([0,1,1])
        vis.add_geometry(y)


        opt = vis.get_render_option()
        # opt.show_coordinate_frame = False
        vis.run()
        vis.destroy_window()