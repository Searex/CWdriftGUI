import numpy as np
import matplotlib.pyplot as plt 
import tkinter as tk

########################################################
##################INITIALIZATION########################
########################################################

# simulation resolution
time_resolution = 1.0;

# y axis inversion tag
invert = 0;

# target orbital parameters
radius_Earth = 6378100.;  # radius of Earth [m]
radius_Moon = 1740000.; # radius of Moon [m]
muEarth = 398600441800000.;    # gravitational parameter for Earth
muMoon = 4904869590000.;    # gravitational parameter for Earth
GUI_altitude_initial = 400000.0;
GUI_planetary_initial = 1;

# initial text values for GUI
GUI_orbit_initial = 'LEO.txt'
GUI_x_initial = 0.00;
GUI_y_initial = 0.00;
GUI_z_initial = 0.00;
GUI_Vx_initial = 0.00;
GUI_Vy_initial = 0.00;
GUI_Vz_initial = 0.00;
GUI_periods = 1.0;

# initialize before accepting user inputs
x_i = 0.0;    # initial relative radial position (relative altitude)
y_i = 0.0;    # initial relative position along orbital path
z_i = 0.0;    # initial relative out-of-plane position
Vx_i = 0.0;   # initial relative radial velocity
Vy_i = 0.0;   # initial relative velocity along orbital path
Vz_i = 0.0;   # initial relative out-of-plane velocity

# initialize storage space for final
x_f = 0.0; # final relative x position storage
y_f = 0.0; # final relative y position storage
z_f = 0.0; # final relative z position storage

# initialize data arrays
state_t = 1;    # time data storage
state_x = 1;    # x data storage
state_y = 1;    # y data storage
state_z = 1;    # z data storage
state_Vx = 1;   # Vx data storage
state_Vy = 1;   # Vy data storage
state_Vz = 1;   # Vz data storage
state_sep = 1;  # minimum separation from KOZ

########################################################
####################LOAD#FILES##########################
########################################################

# button to get filename and load orbital parameter (default is 400km LEO)
def load_orbit():
    global n, mu
    altitude = float(ent_altitude.get()) # target altitude
    planetary = float(ent_planetary.get()) # central body
    if planetary == 1:
        mu = muEarth;
        radius = radius_Earth + altitude;
    elif planetary == 2:
        mu = muMoon;
        radius = radius_Moon + altitude;
    n = np.sqrt(mu/radius**3.);  # orbital parameter
    print("n = ", n)

########################################################
####################CALCULATION#########################
########################################################

def record_states_and_calc():
    # collect initial conditions
    global x_i, y_i, z_i, Vx_i, Vy_i, Vz_i
    load_orbit()
    x_i = float(ent_x_initial.get())
    y_i = float(ent_y_initial.get())
    z_i = float(ent_z_initial.get())
    Vx_i = float(ent_Vx_initial.get())
    Vy_i = float(ent_Vy_initial.get())
    Vz_i = float(ent_Vz_initial.get())
    periods = float(ent_periods.get())

    # time and resolution parameters
    dt = time_resolution; # time step for data points
    time_final = periods * 2. * 3.1415926 / n;

    state_0 = [x_i, y_i, z_i, Vx_i, Vy_i, Vz_i];
    State_log = Calculate_segment(state_0, time_final, dt)

    # export state to global variables
    global state_t, state_x, state_y, state_z, state_Vx, state_Vy, state_Vz
    state_x = State_log[:,0];
    state_y = State_log[:,1];
    state_z = State_log[:,2];
    state_Vx = State_log[:,3];
    state_Vy = State_log[:,4];
    state_Vz = State_log[:,5];
    steps = int(time_final/dt)+1;
    time = 0;
    time_log = np.zeros(steps);
    for i in range(0,steps):
        time_log[i] = time;
        time = time + dt;
    state_t = time_log;
    V_arrival = np.sqrt(state_Vx[steps-1] ** 2. + state_Vy[steps-1] ** 2. + state_Vz[steps-1] ** 2.);
    global x_f, y_f, z_f
    x_f = state_x[-1];
    y_f = state_y[-1];
    z_f = state_z[-1];
    print("final relative x position = ", state_x[-1])
    print("final relative y position = ", state_y[-1])
    print("final relative z position = ", state_z[-1])
    print("final relative x velocity = ", state_Vx[-1])
    print("final relative y velocity = ", state_Vy[-1])
    print("final relative z velocity = ", state_Vz[-1])
    print("final relative velocity = ", V_arrival)
    print("------------------------------------------")

    #print_results()


########################################################
####################PROPOGATION#########################
########################################################

def Calculate_path(segments):
    burnCommands = burns;
    time, Vx_0, Vy_0, Vz_0 = burnCommands[0,:];
    state_0 = [x_i, y_i, z_i, Vx_0, Vy_0, Vz_0]
    State_log = Calculate_segment(state_0, time)
    x = State_log[len(State_log)-1,0]
    y = State_log[len(State_log)-1,1]
    z = State_log[len(State_log)-1,2]
    Vx = State_log[len(State_log)-1,3]
    Vy = State_log[len(State_log)-1,4]
    Vz = State_log[len(State_log)-1,5]
    for i in range(1, segments):
        state_0 = [x, y, z, Vx + burnCommands[i,1], Vy + burnCommands[i,2], Vz + burnCommands[i,3]];
        time = burnCommands[i,0];
        State_log = np.concatenate((State_log,Calculate_segment(state_0, time)))
        x = State_log[len(State_log)-1,0]
        y = State_log[len(State_log)-1,1]
        z = State_log[len(State_log)-1,2]
        Vx = State_log[len(State_log)-1,3]
        Vy = State_log[len(State_log)-1,4]
        Vz = State_log[len(State_log)-1,5]
    return State_log

def Calculate_segment(state_0, time_final, dt):
    steps = int(round(time_final/dt))+1;              # determinine time steps to target
    State_log = np.zeros((steps,6));                # initialize data storage
    time = 0;                                       # initialize segment time
    for i in range(0,steps):
        phi = Phi_make(time);                        # make phi
        State_log[i,:] = np.matmul(phi, state_0);   # get new state
        time = time+dt;                             # step time
    return State_log

########################################################
#####################PHI#MATRIX#########################
########################################################

def Phi_make(t):
    # make phi matrix for no thrust part of solution
    nt = n*t;
    cos = np.cos(nt);
    sin = np.sin(nt);
    phi_0 = np.array([4.-3.*cos, 0., 0., sin/n, (2.-2.*cos)/n, 0.]);                    # x
    phi_1 = np.array([6.*(sin-nt), 1., 0., (2.*cos-2.)/n, (4.*sin-3.*nt)/n, 0.]);       # y
    phi_2 = np.array([0., 0., cos, 0, 0, sin/n]);                                       # z
    phi_3 = np.array([3.*n*sin, 0., 0., cos, (2.*sin), 0.]);                            # Vx
    phi_4 = np.array([6.*n*(cos-1.), 0., 0., -2.*sin, -3.+4.*cos, 0.]);                 # Vy
    phi_5 = np.array([0., 0., -1.*n*sin, 0., 0., cos]);                                 # Vz
    phi = np.array([phi_0, phi_1, phi_2, phi_3, phi_4, phi_5]);
    return phi

########################################################
#######################PLOTS############################
########################################################

# Plot trajectory
def plot_LVLH():
    global invert
    plt.figure(1)
    plt.plot(state_y, state_x)
    plt.plot(y_i, x_i, '^k')
    plt.plot(y_f, x_f, 'sk')
    #plt.xlim(-10, 10)
    #plt.ylim(-10, 10)
    if invert == 0:
        plt.gca().invert_xaxis()
        invert = 1;
        plt.grid()
    plt.title('Local Vertical - Local Horizontal View, triangle = initial, square = final')
    plt.xlabel('y axis (<- prograde) [m]')
    plt.ylabel('x axis (<- nadir) [m]')
    plt.show()

def plot_above():
    plt.figure(2)
    plt.plot(state_z, state_y)
    plt.plot(z_i, y_i, '^k')
    plt.plot(z_f, y_f, 'sk')
    #plt.xlim(-10, 10)
    #plt.ylim(-10, 10)
    plt.title('View from Above, triangle = initial, square = final')
    plt.xlabel('z axis (<- port of target) [m]')
    plt.ylabel('y axis (<- retrograde) [m]')
    plt.grid()
    plt.show()

def plot_behind():
    plt.figure(3)
    plt.plot(state_z, state_x)
    plt.plot(z_i, x_i, '^k')
    plt.plot(z_f, x_f, 'sk')
    #plt.xlim(-10, 10)
    #plt.ylim(-10, 10)
    plt.title('View from Behind, triangle = initial, square = final')
    plt.xlabel('z axis (<- port of target) [m]')
    plt.ylabel('x axis (<- nadir) [m]')
    plt.grid()
    plt.show()

def close_LVLH():
    plt.close(1)

def close_above():
    plt.close(2)

def close_behind():
    plt.close(3)

def close_all():
    plt.close('all')


########################################################
########################GUI#############################
########################################################

# initialize window
window = tk.Tk()
window.title("Relative Nav Coast Test")

frm_input = tk.Frame(master=window)

# title
lbl_title = tk.Label(master=frm_input, text = "Clohessy-Wiltshire Relative Orbital Trajectory Calculator")

# load orbital parameters
lbl_altitude = tk.Label(master=frm_input, text = "Target altitude [m] =")
lbl_planetary = tk.Label(master=frm_input, text = "Central body (1 = Earth, 2 = Moon)")
ent_altitude = tk.Entry(master=frm_input, width=10)
ent_planetary = tk.Entry(master=frm_input, width=10)
ent_altitude.insert(0, GUI_altitude_initial)
ent_planetary.insert(0, GUI_planetary_initial)

# initial position
lbl_x_initial = tk.Label(master=frm_input, text = "x initial [m] =")
lbl_y_initial = tk.Label(master=frm_input, text = "y initial [m] =")
lbl_z_initial = tk.Label(master=frm_input, text = "z initial [m] =")
ent_x_initial = tk.Entry(master=frm_input, width=10)
ent_y_initial = tk.Entry(master=frm_input, width=10)
ent_z_initial = tk.Entry(master=frm_input, width=10)
ent_x_initial.insert(0, GUI_x_initial)
ent_y_initial.insert(0, GUI_y_initial)
ent_z_initial.insert(0, GUI_z_initial)

# initial velocity
lbl_Vx_initial = tk.Label(master=frm_input, text = "Vx initial [m/s]=")
lbl_Vy_initial = tk.Label(master=frm_input, text = "Vy initial [m/s]=")
lbl_Vz_initial = tk.Label(master=frm_input, text = "Vz initial [m/s]=")
ent_Vx_initial = tk.Entry(master=frm_input, width=10)
ent_Vy_initial = tk.Entry(master=frm_input, width=10)
ent_Vz_initial = tk.Entry(master=frm_input, width=10)
ent_Vx_initial.insert(0, GUI_Vx_initial)
ent_Vy_initial.insert(0, GUI_Vy_initial)
ent_Vz_initial.insert(0, GUI_Vz_initial)

# duration of simulation in periods
lbl_periods = tk.Label(master=frm_input, text = "Duration of simulation [periods] = ")
ent_periods = tk.Entry(master=frm_input, width=10)
ent_periods.insert(0, GUI_periods)

# calculate button
btn_calculate = tk.Button(master=frm_input, text="Calculate", command=record_states_and_calc)

# plot buttons
btn_plot_lvlh = tk.Button(master=frm_input, text="Add to LVLH view", command=plot_LVLH)
btn_plot_above = tk.Button(master=frm_input, text="Add to above view", command=plot_above)
btn_plot_behind = tk.Button(master=frm_input, text="Add to behind view", command=plot_behind)

# close plot buttons
btn_plot_lvlh_off = tk.Button(master=frm_input, text="Close LVLH view", command=close_LVLH)
btn_plot_above_off = tk.Button(master=frm_input, text="Close above view", command=close_above)
btn_plot_behind_off = tk.Button(master=frm_input, text="Close behind view", command=close_behind)
btn_plot_all_off = tk.Button(master=frm_input, text="Close all views", command=close_all)

# window formatting
frm_input.grid(row=0, column=0, pady=10)

lbl_title.grid(row=1, column=1, padx=0, pady=20)

lbl_altitude.grid(row=3, column=0, padx=10, sticky="e")
ent_altitude.grid(row=3, column=1, padx=0, sticky="w")
lbl_planetary.grid(row=4, column=0, padx=10, sticky="e")
ent_planetary.grid(row=4, column=1, padx=0, sticky="w")

lbl_x_initial.grid(row=10, column=0, padx=10, sticky="e")
ent_x_initial.grid(row=10, column=1, padx=0, sticky="w")
lbl_y_initial.grid(row=11, column=0, padx=10, sticky="e")
ent_y_initial.grid(row=11, column=1, padx=0, sticky="w")
lbl_z_initial.grid(row=12, column=0, padx=10, sticky="e")
ent_z_initial.grid(row=12, column=1, padx=0, sticky="w")

lbl_Vx_initial.grid(row=10, column=1, padx=0, sticky="e")
ent_Vx_initial.grid(row=10, column=2, padx=0, sticky="w")
lbl_Vy_initial.grid(row=11, column=1, padx=0, sticky="e")
ent_Vy_initial.grid(row=11, column=2, padx=0, sticky="w")
lbl_Vz_initial.grid(row=12, column=1, padx=0, sticky="e")
ent_Vz_initial.grid(row=12, column=2, padx=0, sticky="w")

lbl_periods.grid(row=13, column=1, padx=0, sticky="e")
ent_periods.grid(row=13, column=2, padx=0, sticky="w")

btn_calculate.grid(row=15, column=1, padx=0, pady=0, sticky="e")

btn_plot_lvlh.grid(row=21, column=1, padx=0, pady=0, sticky="w")
btn_plot_above.grid(row=22, column=1, padx=0, pady=0, sticky="w")
btn_plot_behind.grid(row=23, column=1, padx=0, pady=0, sticky="w")

btn_plot_lvlh_off.grid(row=21, column=2, padx=0, pady=0, sticky="w")
btn_plot_above_off.grid(row=22, column=2, padx=0, pady=0, sticky="w")
btn_plot_behind_off.grid(row=23, column=2, padx=0, pady=0, sticky="w")
btn_plot_all_off.grid(row=24, column=2, padx=0, pady=0, sticky="w")

window.mainloop()
