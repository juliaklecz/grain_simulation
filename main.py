import tkinter as tk
import numpy as np
from collections import Counter
import random
import time
import math
from PIL import Image, ImageDraw


# Constants
IMAGE_SIZE = (150, 150, 150)
WHITE_COLOR = "#FFFFFF"
NEIGHBORHOOD_OPTIONS = ["Von Neumann", "Moore", "Hexagonal"]
BOUNDARY_OPTIONS = ["Absorbing", "Periodic"]
UPDATE_INTERVAL = 100  # in milliseconds
START_OPTIONS = ["Random", "Intervals", "Manual"]

# Global variables
image_np = np.full((0, 0, 0), WHITE_COLOR)
COLORS = []
COLORS_INDEX = 0
SIMULATION_GOING = False
MONTE_CARLO_GOING = False
current_start_option = START_OPTIONS[0]
current_iteration = 0
max_iterations = None
filename = "result.txt"
start_time = None
stop_time = None
DELETED_GRAINS = []


# GUI functions

def create_canvas():
    global IMAGE_SIZE, image_np, canvas
    width = int(canvas_width.get())
    height = int(canvas_height.get())
    depth = int(canvas_depth.get())
    IMAGE_SIZE = (width, height, depth)
    image_np = np.full((IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2]), WHITE_COLOR)
    canvas.config(width=IMAGE_SIZE[1] * 5, height=IMAGE_SIZE[0] * 5)
    draw_image()


def start_simulation():
    global SIMULATION_GOING, current_iteration, max_iterations, start_time
    SIMULATION_GOING = True
    current_iteration = 0
    max_iterations = None
    start_time = time.time()
    if iterations_var.get().strip().isdigit():
        max_iterations = int(iterations_var.get().strip())
    grow_grains()


def start_monte_carlo():
    global MONTE_CARLO_GOING
    MONTE_CARLO_GOING = True
    update_monte_carlo()


def stop_simulation():
    global SIMULATION_GOING, MONTE_CARLO_GOING, stop_time
    SIMULATION_GOING = False
    MONTE_CARLO_GOING = False
    stop_time = time.time()


# noinspection PyTypeChecker
def update_image():
    global canvas, image_np, SIMULATION_GOING, current_iteration, max_iterations
    if SIMULATION_GOING:
        current_iteration += 1
        if max_iterations is not None and current_iteration >= max_iterations:
            stop_simulation()
    draw_image()
    if SIMULATION_GOING:
        canvas.after(UPDATE_INTERVAL, update_image)


def adjust_coordinates(x, y):
    canvas_center_x = canvas.winfo_width() / 2
    canvas_center_y = canvas.winfo_height() / 2
    adjusted_x = int((x - canvas_center_x) / 5 + IMAGE_SIZE[1] / 2)
    adjusted_y = int((y - canvas_center_y) / 5 + IMAGE_SIZE[0] / 2)
    return adjusted_x, adjusted_y


def reset_image(start_option):
    global image_np, current_iteration, MONTE_CARLO_GOING
    current_iteration = 0
    MONTE_CARLO_GOING = False
    if start_option == "Random":
        image_np = random_start()
    elif start_option == "Intervals":
        image_np = intervals_start()
    elif start_option == "Manual":
        image_np = np.full((IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2]), WHITE_COLOR)
    draw_image()


def manual_start(event):
    if delete_checkbox_var.get() == 1:
        delete_grain(event)
    else:
        global image_np
        if start_option_var.get() == "Manual":
            x, y = adjust_coordinates(event.x, event.y)
            color = generate_unique_color()
            image_np[y, x, 0] = color  # Set color on the front face (z=0)
            draw_image()  # Reflect the manual change immediately


def delete_grain(event):
    global image_np, COLORS, DELETED_GRAINS
    x, y = adjust_coordinates(event.x, event.y)
    target_color = image_np[y, x, 0]  # Check on the front face (z=0)

    if target_color == WHITE_COLOR:
        return  # Nothing to delete if the clicked pixel is already white

    # Flood-fill algorithm to delete the grain
    def flood_fill(image, xff, yff, z, target_color_ff, replacement_color):
        stack = [(xff, yff, z)]
        while stack:
            cx, cy, cz = stack.pop()
            if image[cy, cx, cz] == target_color_ff:
                image[cy, cx, cz] = replacement_color
                if cx > 0:
                    stack.append((cx - 1, cy, cz))
                if cx < IMAGE_SIZE[1] - 1:
                    stack.append((cx + 1, cy, cz))
                if cy > 0:
                    stack.append((cx, cy - 1, cz))
                if cy < IMAGE_SIZE[0] - 1:
                    stack.append((cx, cy + 1, cz))
                if cz > 0:
                    stack.append((cx, cy, cz - 1))
                if cz < IMAGE_SIZE[2] - 1:
                    stack.append((cx, cy, cz + 1))

    flood_fill(image_np, x, y, 0, target_color, WHITE_COLOR)

    # Remove the color from the COLORS list
    if target_color in COLORS:
        DELETED_GRAINS.append(target_color)
        COLORS.remove(target_color)

    draw_image()


def draw_image():
    global canvas, image_np
    canvas.delete("all")
    canvas_center_x = canvas.winfo_width() / 2
    canvas_center_y = canvas.winfo_height() / 2
    image_top_left_x = canvas_center_x - (IMAGE_SIZE[1] * 5 / 2)
    image_top_left_y = canvas_center_y - (IMAGE_SIZE[0] * 5 / 2)
    z = 0  # Visualize the front face
    for i in range(IMAGE_SIZE[0]):
        for j in range(IMAGE_SIZE[1]):
            color = image_np[i, j, z]
            canvas.create_rectangle(
                image_top_left_x + j * 5,
                image_top_left_y + i * 5,
                image_top_left_x + (j + 1) * 5,
                image_top_left_y + (i + 1) * 5,
                fill=color,
                outline=""
            )


# noinspection PyTypeChecker
def grow_grains():
    global SIMULATION_GOING, current_iteration, max_iterations
    if SIMULATION_GOING:
        new_image = np.copy(image_np)
        for x in range(IMAGE_SIZE[0]):
            for y in range(IMAGE_SIZE[1]):
                for z in range(IMAGE_SIZE[2]):
                    if image_np[x, y, z] == WHITE_COLOR:
                        neighbors = get_neighbors(x, y, z)
                        if neighbors:
                            # Find the most frequent color among neighbors
                            most_common_color = Counter(neighbors).most_common(1)[0][0]
                            new_image[x, y, z] = most_common_color
        image_np[:, :, :] = new_image
        draw_image()

        # Calculate current coverage
        current_coverage = calculate_coverage()
        target_coverage = float(target_coverage_var.get().strip())

        current_iteration += 1
        if (max_iterations is None or current_iteration < max_iterations) and current_coverage < target_coverage:
            canvas.after(UPDATE_INTERVAL, grow_grains)
        else:
            stop_simulation()


def calculate_coverage():
    total_cells = np.product(IMAGE_SIZE)
    filled_cells = np.count_nonzero(image_np != WHITE_COLOR)
    return (filled_cells / total_cells) * 100


def update_monte_carlo():
    global canvas, image_np, MONTE_CARLO_GOING
    if MONTE_CARLO_GOING:
        monte_carlo_step()
        draw_image()
        canvas.after(UPDATE_INTERVAL, update_monte_carlo)


def monte_carlo_step():
    global image_np
    for _ in range(IMAGE_SIZE[0] * IMAGE_SIZE[1] * IMAGE_SIZE[2]):
        x = random.randint(0, IMAGE_SIZE[0] - 1)
        y = random.randint(0, IMAGE_SIZE[1] - 1)
        z = random.randint(0, IMAGE_SIZE[2] - 1)
        neighbors = get_neighbors(x, y, z)
        if neighbors:
            neighbor_color = random.choice(neighbors)
            if should_change_state(x, y, z, neighbor_color):
                image_np[x, y, z] = neighbor_color


def should_change_state(x, y, z, neighbor_color):
    current_color = image_np[x, y, z]
    if current_color == neighbor_color:
        return False

    # Calculate current energy
    current_energy = calculate_energy(x, y, z, current_color)

    # Calculate new energy if the state changes to neighbor_color
    new_energy = calculate_energy(x, y, z, neighbor_color)

    # Energy difference
    delta_energy = new_energy - current_energy

    # Decide whether to change state
    if delta_energy < 0:
        return True  # Favorable change
    else:
        # Use a probabilistic acceptance criterion
        kt = 0.1
        probability = math.exp(-delta_energy / kt)
        return random.random() < probability


def calculate_energy(x, y, z, color):
    energy = 0
    neighbors = get_neighbors(x, y, z)
    for neighbor_color in neighbors:
        if neighbor_color != color:
            energy += 1  # Increase energy for each differing neighbor
    return energy


def get_neighbors(x, y, z):
    neighbors = []
    if neighborhood_option_var.get() == "Von Neumann":
        directions = [
            (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0),  # 2D Neighbors
            (0, 0, 1), (0, 0, -1)  # Add Z dimension neighbors
        ]
    elif neighborhood_option_var.get() == "Moore":
        directions = [
            (dx, dy, dz)
            for dx in [-1, 0, 1]
            for dy in [-1, 0, 1]
            for dz in [-1, 0, 1]
            if (dx, dy, dz) != (0, 0, 0)
        ]
    elif neighborhood_option_var.get() == "Hexagonal":
        directions = [
            (-1, -1, 0), (-1, 0, 0), (0, -1, 0), (1, 1, 0), (1, 0, 0), (0, 1, 0),  # 2D Hexagonal neighbors
            (-1, 0, -1), (1, 0, -1), (0, -1, -1), (0, 1, -1),  # 3D Hexagonal neighbors for other layers
            (-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1)
        ]
    else:
        directions = []

    for dx, dy, dz in directions:
        nx, ny, nz = x + dx, y + dy, z + dz
        nx, ny, nz = apply_boundary(nx, ny, nz)
        if image_np[nx, ny, nz] != WHITE_COLOR:
            neighbors.append(image_np[nx, ny, nz])
    return neighbors


def apply_boundary(x, y, z):
    if boundary_option_var.get() == "Absorbing":
        new_x = max(0, min(x, IMAGE_SIZE[0] - 1))
        new_y = max(0, min(y, IMAGE_SIZE[1] - 1))
        new_z = max(0, min(y, IMAGE_SIZE[2] - 1))
        return new_x, new_y, new_z
    elif boundary_option_var.get() == "Periodic":
        x = x % IMAGE_SIZE[0]
        y = y % IMAGE_SIZE[1]
        z = z % IMAGE_SIZE[2]
    return x, y, z


def random_start():
    image = np.full((IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2]), WHITE_COLOR)
    num_grains = int(num_grains_entry.get().strip())
    positions = set()
    while len(positions) < num_grains:
        x = random.randint(0, IMAGE_SIZE[0] - 1)
        y = random.randint(0, IMAGE_SIZE[1] - 1)
        z = random.randint(0, IMAGE_SIZE[2] - 1)
        positions.add((x, y, z))
    for x, y, z in positions:
        image[x, y, z] = generate_unique_color()
    return image


def intervals_start():
    global COLORS
    COLORS = [generate_unique_color() for _ in range(10)]
    intervals_image = np.full(IMAGE_SIZE, WHITE_COLOR)
    interval = max(1, IMAGE_SIZE[0] // len(COLORS))
    for idx, color in enumerate(COLORS):
        intervals_image[idx * interval:(idx + 1) * interval, :, :] = color
    return intervals_image


def generate_unique_color():
    global COLORS
    while True:
        color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        if color not in COLORS:
            COLORS.append(color)
            return color


def export_to_xyz():
    global filename
    with open(filename, 'w') as f:
        # Write the number of atoms (or grains) as the first line
        f.write(str(np.count_nonzero(image_np)) + '\n')

        # Write a comment line
        f.write("Generated by your_simulation_script.py\n")

        # Write the coordinates and IDs of each grain in XYZ format
        id_counter = 1
        id_map = {}  # Map colors to IDs

        for z in range(IMAGE_SIZE[2]):
            for y in range(IMAGE_SIZE[1]):
                for x in range(IMAGE_SIZE[0]):
                    if image_np[x, y, z] != WHITE_COLOR:
                        color = image_np[x, y, z]
                        if color not in id_map:
                            id_map[color] = id_counter
                            id_counter += 1

                        grain_id = id_map[color]
                        f.write(" ".join(map(str, [x, y, z, grain_id])) + "\n")
                    elif image_np[x, y, z] == WHITE_COLOR:
                        grain_id = 0


                        f.write(" ".join(map(str, [x, y, z, grain_id])) + "\n")


def normalize_image():
    global image_np
    # Set all non-white pixels to red
    red_color = "#FF0000"
    image_np[image_np != WHITE_COLOR] = red_color
    draw_image()


# noinspection PyUnresolvedReferences,PyTypeChecker
def show_stats():
    # Calculate statistics
    initial_grains = num_grains_entry.get()
    current_grains = np.count_nonzero(image_np != WHITE_COLOR)
    current_pores = np.count_nonzero(image_np == WHITE_COLOR)
    deleted_grains = len(DELETED_GRAINS)
    grain_to_pore_ratio = current_grains / current_pores if current_pores != 0 else "N/A"
    porosity = np.count_nonzero(image_np == WHITE_COLOR) / np.product(IMAGE_SIZE)
    # Calculate time taken
    elapsed_time = 0
    if start_time is not None:
        elapsed_time = time.time() - start_time if stop_time is None else stop_time - start_time

    # Create a new window to display statistics
    stats_window = tk.Toplevel(root)
    stats_window.title("Statistics")
    # Populate the window with labels to display statistics
    tk.Label(stats_window, text="Initial Grains: {}".format(initial_grains)).pack()
    tk.Label(stats_window, text="Deleted Grains: {}".format(deleted_grains)).pack()
    tk.Label(stats_window, text="Grain to Pore Ratio: {}".format(grain_to_pore_ratio)).pack()
    tk.Label(stats_window, text="Porosity: {:.2f}%".format(porosity * 100)).pack()
    tk.Label(stats_window, text="Time Taken: {:.2f} seconds".format(elapsed_time)).pack()


def visualize_energy(pixel_color, x, y, z):
    # Get the colors of neighboring pixels
    neighbors = get_neighbors(x, y, z)

    # Check if any neighbor has a different color than the current pixel
    for neighbor_color in neighbors:
        if neighbor_color != pixel_color:
            # If different color found, return a color to represent the edge
            return "red"  # You can choose any color you like to represent the edge

    # If all neighbors have the same color, return the color of the pixel
    return pixel_color


def update_image_with_energy():
    global canvas, image_np
    canvas.delete("all")
    canvas_center_x = canvas.winfo_width() / 2
    canvas_center_y = canvas.winfo_height() / 2
    image_top_left_x = canvas_center_x - (IMAGE_SIZE[1] * 5 / 2)
    image_top_left_y = canvas_center_y - (IMAGE_SIZE[0] * 5 / 2)
    z = 0  # Visualize the front face
    for i in range(IMAGE_SIZE[0]):
        for j in range(IMAGE_SIZE[1]):
            color = image_np[i, j, z]
            # Visualize the energy by checking the pixel's neighbors
            color = visualize_energy(color, i, j, z)
            canvas.create_rectangle(
                image_top_left_x + j * 5,
                image_top_left_y + i * 5,
                image_top_left_x + (j + 1) * 5,
                image_top_left_y + (i + 1) * 5,
                fill=color,
                outline=""
            )


def save_canvas_as_png():

    file_path = "result_image.png"

    # Create a blank image with white background
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()
    image = Image.new("RGB", (canvas_width, canvas_height), WHITE_COLOR)
    draw = ImageDraw.Draw(image)

    # Copy the content from the canvas to the image
    for i in range(IMAGE_SIZE[0]):
        for j in range(IMAGE_SIZE[1]):
            color = image_np[i, j, 0]  # We are visualizing the front face (z=0)
            if color != WHITE_COLOR:
                x1 = j * 5
                y1 = i * 5
                x2 = x1 + 5
                y2 = y1 + 5
                draw.rectangle([x1, y1, x2, y2], fill=color, outline=color)

    # Save the image as a PNG file
    image.save(file_path)


# GUI setup
root = tk.Tk()
root.title("Grain Growth Simulation")

main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

# Canvas and settings frame
canvas = tk.Canvas(main_frame, width=IMAGE_SIZE[1] * 5, height=IMAGE_SIZE[0] * 5)
canvas.grid(row=0, column=0, rowspan=20, sticky="nsew")

canvas_settings_frame = tk.Frame(main_frame)
canvas_settings_frame.grid(row=0, column=1, sticky="nw")

canvas_width_label = tk.Label(canvas_settings_frame, text="Width:")
canvas_width_label.grid(row=0, column=0)
canvas_width = tk.Entry(canvas_settings_frame, width=7)
canvas_width.grid(row=0, column=1)
canvas_width.insert(0, "150")

canvas_height_label = tk.Label(canvas_settings_frame, text="Height:")
canvas_height_label.grid(row=1, column=0)
canvas_height = tk.Entry(canvas_settings_frame, width=7)
canvas_height.grid(row=1, column=1)
canvas_height.insert(0, "150")

canvas_depth_label = tk.Label(canvas_settings_frame, text="Depth:")
canvas_depth_label.grid(row=2, column=0)
canvas_depth = tk.Entry(canvas_settings_frame, width=7)
canvas_depth.grid(row=2, column=1)
canvas_depth.insert(0, "1")

iterations_label = tk.Label(canvas_settings_frame, text="Iterations:")
iterations_label.grid(row=3, column=0)
iterations_var = tk.StringVar()
iterations_entry = tk.Entry(canvas_settings_frame, textvariable=iterations_var, width=7)
iterations_entry.grid(row=3, column=1)


tk.Label(canvas_settings_frame, text="Target Coverage (%):").grid(row=4, column=0, sticky="e")
target_coverage_var = tk.Entry(canvas_settings_frame)
target_coverage_var.grid(row=4, column=1)

neighborhood_option_var = tk.StringVar(value=NEIGHBORHOOD_OPTIONS[0])
neighborhood_menu = tk.OptionMenu(canvas_settings_frame, neighborhood_option_var, *NEIGHBORHOOD_OPTIONS)
neighborhood_label = tk.Label(canvas_settings_frame, text="Neighborhood:")
neighborhood_label.grid(row=5, column=0)
neighborhood_menu.grid(row=5, column=1)

boundary_option_var = tk.StringVar(value=BOUNDARY_OPTIONS[0])
boundary_menu = tk.OptionMenu(canvas_settings_frame, boundary_option_var, *BOUNDARY_OPTIONS)
boundary_label = tk.Label(canvas_settings_frame, text="Boundary:")
boundary_label.grid(row=6, column=0)
boundary_menu.grid(row=6, column=1)

start_option_var = tk.StringVar(value=START_OPTIONS[0])
start_menu = tk.OptionMenu(canvas_settings_frame, start_option_var, *START_OPTIONS)
start_label = tk.Label(canvas_settings_frame, text="Start Option:")
start_label.grid(row=7, column=0)
start_menu.grid(row=7, column=1)

num_grains_label = tk.Label(canvas_settings_frame, text="Number of Grains:")
num_grains_label.grid(row=8, column=0)
num_grains_entry = tk.Entry(canvas_settings_frame, width=7)
num_grains_entry.grid(row=8, column=1)
num_grains_entry.insert(0, "100")

interval_label = tk.Label(canvas_settings_frame, text="Interval:")
interval_label.grid(row=9, column=0)
interval_entry = tk.Entry(canvas_settings_frame, width=7)
interval_entry.grid(row=9, column=1)
interval_entry.insert(0, "10")

# Control frame
control_frame = tk.Frame(main_frame)
control_frame.grid(row=9, column=1, sticky="nw")

create_canvas_button = tk.Button(control_frame, text="Create Canvas", command=create_canvas)
create_canvas_button.pack(side=tk.TOP)

start_simulation_button = tk.Button(control_frame, text="Start Simulation", command=start_simulation)
start_simulation_button.pack(side=tk.TOP)

start_monte_carlo_button = tk.Button(control_frame, text="Start Monte Carlo", command=start_monte_carlo)
start_monte_carlo_button.pack(side=tk.TOP)

stop_button = tk.Button(control_frame, text="Stop", command=stop_simulation)
stop_button.pack(side=tk.TOP)

reset_button = tk.Button(control_frame, text="Reset", command=lambda: reset_image(start_option_var.get()))
reset_button.pack(side=tk.TOP)

delete_checkbox_var = tk.IntVar()
delete_checkbox = tk.Checkbutton(control_frame, text="Delete Mode", variable=delete_checkbox_var)
delete_checkbox.pack(side=tk.TOP)

normalize_button = tk.Button(control_frame, text="Normalize", command=normalize_image)
normalize_button.pack(side=tk.TOP)

stats_button = tk.Button(control_frame, text="Stats", command=show_stats)
stats_button.pack(side=tk.TOP)

save_button = tk.Button(control_frame, text="Save", command=export_to_xyz)
save_button.pack(side=tk.TOP)

save_as_png_button = tk.Button(control_frame, text="Save as PNG", command=save_canvas_as_png)
save_as_png_button.pack(side=tk.TOP)

energy_button = tk.Button(control_frame, text="Show energy", command=update_image_with_energy)
energy_button.pack(side=tk.TOP)

# Adjust canvas expandability
main_frame.grid_rowconfigure(0, weight=1)
main_frame.grid_columnconfigure(0, weight=1)

canvas.bind("<Button-1>", manual_start)

# Initial setup
create_canvas()

# Function to show statistics

root.mainloop()