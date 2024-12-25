import numpy as np
import matplotlib.pyplot as plt

def newton_method(guess, tol=5**-2, iter=50):
    # The function we are working on is y = x^5 + x^2 - x
    for _ in range(iter):
        # I had problems with the final shape, it seems like using a mask will result in the shape get smaller and numpy will just broadcast it into the wrong shape
        func = guess**5 + guess**2 - guess
        deriv = 5 * guess**4 + 2 * guess - 1
        
        # Update guess only where tolerance is not met
        mask = abs(func) > tol
        guess[mask] -= func[mask] / deriv[mask]

    return guess

def generate_mandel_set(x_min, x_max, y_min, y_max, higth, width, tol = 5^-5, iter = 100):

    know_roots = [0, -1.2207, 0.072449, 0.24813 + 0.24813 + 1.0340*1j, 0.24813 - 1.0340*1j] #the known roots, from wolframalpha I did not solve this myself


    real_dim = np.linspace(x_min, x_max, width)
    imag_dim = np.linspace(y_min, y_max, higth)
    x, y = np.meshgrid(real_dim, imag_dim)
    guesses = x + y*1j

    #new_guesses = newton_method(guesses, tol = 5^-5, iter = 100)


    new_guesses = newton_method(guesses)
    print(guesses.shape)
    print(new_guesses.shape)

    closest_roots_index = np.argmin(np.abs(new_guesses[:,:,None] - know_roots), axis = 2)

    return closest_roots_index

closest_roots_index = generate_mandel_set(1, 30, 1, 30, 3000, 3000)

plt.figure(figsize=(10, 10))
plt.imshow(closest_roots_index, extent=[-2, 2, -2, 2], cmap="hsv", origin="lower")
plt.colorbar(label="Root Index")
plt.title("Newton Fractal for $x^5 + x^2 - x$")
plt.xlabel("Re")
plt.ylabel("Im")
plt.show()