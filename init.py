import ffmpeg
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import mpmath
from mpmath import *
from matplotlib.animation import FuncAnimation  
from flask import Flask, render_template
from flask import request

app = Flask(__name__)

@app.route('/')
def hello():
	return render_template('home.html')

get_plot = False

@app.route('/1_wave')
def wave1():
	return render_template("1wave.html")

@app.route('/2_wave')
def wave2():
	return render_template("2wave.html")

@app.route('/shallow_wave')
def shallowwave():
    return render_template("shallow_wave.html")


@app.route('/get_plot', methods = ['GET', 'POST'])

def get_plot():
    if request.method == "POST":
        value_a = request.form['value_a']
        t_max = 15  
        _fps = 15   # Increase the frames per second

        a = float(value_a)  # amplitude

        fig = plt.figure()
        x = np.linspace(-10, 25, 10000)
        axis = plt.axes(xlim=(-100, 100), ylim=(-5, 15))
        line, = axis.plot([], [], lw=3)

        def animate_soliton(amplitude, x_range=(0, 20), num_points=1000, num_steps=2000):  # Increase num_steps
            # Parameters
            dx = (x_range[1] - x_range[0]) / (num_points - 1)
            c = 1  # Soliton speed
            dt = 0.001* dx/c

            # Initial condition - soliton profile
            x_values = np.linspace(x_range[0], x_range[1], num_points)
            u = amplitude * np.exp(-0.25 * (x_values - 10)**2).astype(float)

            # Create a plot
            fig, ax = plt.subplots()
            line, = ax.plot(x_values, u)
            ax.set_title('Soliton Animation')
            ax.set_xlabel('x')
            ax.set_ylabel('amplitude')

            def update(frame):
                nonlocal u

                u_old = u.copy()

                # Update using finite difference method
                for i in range(1, num_points - 1):
                    u[i] = u_old[i] - 6 * u_old[i] * (u_old[i+1] - u_old[i-1]) * dt / (2 * dx) + \
                        c**2 * (u_old[i+1] - 2 * u_old[i] + u_old[i-1]) * dt / dx**2

                line.set_ydata(u)  # Fix the update of ydata

                return line,

            anim = FuncAnimation(fig, update, frames=t_max * _fps, interval=1000/_fps, blit=True)

            anim.save('static/continuousSineWave.mp4', writer='ffmpeg', fps=_fps)

        animate_soliton(a)

        return render_template('1wave.html', get_plot=True, plot_url='static/continuousSineWave.mp4')

    else:
        return render_template('1wave.html')

# def solve_pde(amplitude, current_time):
#     #spilt step fourierr method
#     N = 256
#     x = np.linspace(-10, 10, N)
#     delta_x = x[1] - x[0]
#     delta_k = 2 * np.pi / (N * delta_x)
#     k = np.concatenate((np.arange(0, N//2 + 1) * delta_k, np.arange(-(N//2 - 1), 0) * delta_k))
#     c = 16
#     u = 1/2 * c * (1/np.cosh(np.sqrt(c)/2 * (x + 8)))**2 # initial condition

#     delta_t = 0.1 / N**2
#     tmax = 1.0
#     nmax = round(tmax / delta_t)

#     U = np.fft.fft(u)

#     steps = round(current_time / delta_t)

#     for n in range(1, steps + 1):
#         # first we solve the linear part
#         U = U * np.exp(1j * k**3 * delta_t)
#         # then we solve the non-linear part
#         U = U - delta_t * (3j * k * np.fft.fft(np.real(np.fft.ifft(U**2))))

#     u_final = amplitude * np.real(np.fft.ifft(U))
#     return x, u_final


#potentially allow graphs to exist while changing parameters such that users can compare the differences in the data
#solve 2nd order differential equatins by storing the results in an array and then displaying the array

@app.route('/plot', methods = ['GET', 'POST'])
def plot():
    if request.method == "POST":
        value_a = request.form['value_a']
        value_b = request.form['value_b']

        print(request.form.get('checks'))

        t_max = 15
        _fps = 10

        a = float(value_a) #amplitude
        b = float(value_b) #phase shift

        
        fig = plt.figure()
        # marking the x-axis and y-axis
        x = np.linspace(-10, 25, 10000)
        axis = plt.axes(xlim =(-10,25),  
                ylim =(-1,10))  

        # initializing a line variable 
        line, = axis.plot([], [], lw = 3)  

        if request.form.get('checks') == 'checked':
            print(a,b)

            line1, = axis.plot([], [], lw=3, label = "wave 1")
            line2, = axis.plot([], [], lw=3, label = "wave 2")
            line3, = axis.plot([], [], lw=3, label = "combined wave")

            def init():  
                line1.set_data([], []) 
                line2.set_data([], [])
                line3.set_data([], [])
                return line1, line2, line3,

            def animate(i):
            
                #a(sin(bx+c))+d
                # y = a*np.sin(b * np.pi * (x - c * i)) + d
                # y = np.sin(2*np.pi*(x-0.1*i))
                # fix cases; negative numbers and floats
                # y = a*np.sin(b*np.pi*(x- c*i))+d
                # y = a*np.square(1/np.cosh((float(sqrt(a/2))*(x-2*a*i-b))))

                # η = x - a * i
                # y = (a/2) / np.cosh(np.sqrt(a) *  η/ 2) ** 2

                x1 = -5+a * i/20
                x2 = 5+b * i/20

                # Calculate the positions of the solitons
                # Calculate the energy of the waves and show that the energy is constant
                # allow the users to show the orriginal solitons
                # show how linear waves interract and comapre the two
                # compare numerical solution to analytical solution

                #numerical solution via package/numerical code



                #analytical solution

                η1 = x - x1
                η2 = x - x2
                u1 = (a / 2) / np.cosh(np.sqrt(a) * η1 / 2) ** 2
                u2 = (b / 2) / np.cosh(np.sqrt(b) * η2 / 2) ** 2
                y = u1 + u2
                line1.set_data(x, y)
                line2.set_data(x, u1)
                line3.set_data(x, u2)

                return line1, line2, line3,

                # Show the plot
            anim = FuncAnimation(fig, animate, frames = t_max*_fps, interval = 1000, blit = True) 

            anim.save('static/2wave3waves.mp4', writer = 'ffmpeg', fps = _fps) 
            
            return render_template('2wave.html', get_plot = True, plot_url = 'static/2wave3waves.mp4')

        else:
            print(a,b)

            def init1():  
                line.set_data([], []) 
                return line, 

            def animate1(i):
            
                #a(sin(bx+c))+d
                # y = a*np.sin(b * np.pi * (x - c * i)) + d
                # y = np.sin(2*np.pi*(x-0.1*i))
                # fix cases; negative numbers and floats
                # y = a*np.sin(b*np.pi*(x- c*i))+d
                # y = a*np.square(1/np.cosh((float(sqrt(a/2))*(x-2*a*i-b))))

                # η = x - a * i
                # y = (a/2) / np.cosh(np.sqrt(a) *  η/ 2) ** 2

                x1 = -5+a * i/20
                x2 = 5+b * i/20

                # Calculate the positions of the solitons
                # Calculate the energy of the waves and show that the energy is constant
                # allow the users to show the orriginal solitons
                # show how linear waves interract and comapre the two
                # compare numerical solution to analytical solution

                #numerical solution via package/numerical code

                #analytical solution
                η1 = x - x1
                η2 = x - x2
                u1 = (a / 2) / np.cosh(np.sqrt(a) * η1 / 2) ** 2
                u2 = (b / 2) / np.cosh(np.sqrt(b) * η2 / 2) ** 2
                y = u1 + u2
                line.set_data(x,y)
                return line,

            # Show the plot
            anim = FuncAnimation(fig, animate1, init_func = init1, frames = t_max*_fps, interval = 1000, blit = True) 

            anim.save('static/2wavesuper.mp4', writer = 'ffmpeg', fps = _fps) 
        
            return render_template('2wave.html', get_plot = True, plot_url = 'static/2wavesuper.mp4')

    else:
        return render_template('2wave.html')


@app.route('/shallow', methods=['GET', 'POST'])
def shallow():
    if request.method == 'POST':
        # Get user inputs from the form
        depth = float(request.form['depth']) #initial depth
        amplitude = float(request.form['amplitude'])
        speed = float(request.form['speed'])

        t_max = 30
        _fps = 10

        fig = plt.figure()
        x = np.linspace(-10, 25, 10000)
        axis = plt.axes(xlim =(-10,25),  
            ylim =(-10,10))  

        line, = axis.plot([], [], lw = 3)  

        water_depth = 1.0 + 0.5 * np.sin(2 * np.pi * x / 25) # adding variability to water depth via a sin function

        def init():  
            line.set_data([], []) 
            return line, 

        def animate(i):
            x1 = -5 + speed * i / 20
            η1 = x - x1
            a1 = np.sqrt(amplitude)
            u1 = a1 / np.cosh(a1 * η1 / 2) ** 2

            # Incorporate water depth
            y = u1 - depth - water_depth

            # Introduce dispersion (linear term)
            dispersion_term = 0.1 * np.gradient(np.gradient(y,x),x)

            y_dispersion =(y + dispersion_term) # taking the absolute value inverts the soliton, allows elevation to be above zero
            line.set_data(x, y_dispersion + depth)
            return line,

        anim = FuncAnimation(fig, animate, init_func = init, frames = t_max*_fps, interval = 5000, blit = True)

        anim.save('static/shallowwave.mp4', writer='ffmpeg', fps=_fps)

        return render_template('shallow_wave.html', get_plot = True, plot_url = 'static/shallowwave.mp4')

    else:
        return render_template('sshallow_wave.html')



app.secret_key = 'some key that you will never guess'

#Run the app on localhost port 5000
if __name__ == "__main__":
    app.run('127.0.0.1', 5000, debug = True)

