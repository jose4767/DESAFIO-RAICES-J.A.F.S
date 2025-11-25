import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# Configuración de estilo
plt.style.use('default')

# DEFINICIÓN DE LAS FUNCIONES
def f1(x):
    return x**3 - np.exp(0.8*x) - 20

def df1(x):
    return 3*x**2 - 0.8*np.exp(0.8*x)

def f2(x):
    return 3*np.sin(0.5*x) - 0.5*x + 2

def df2(x):
    return 1.5*np.cos(0.5*x) - 0.5

def f3(x):
    return x**3 - x**2*np.exp(-0.5*x) - 3*x + 1

def df3(x):
    return 3*x**2 + np.exp(-0.5*x)*(0.5*x**2 - 2*x) - 3

def f4(x):
    return np.cos(x)**2 - 0.5*x*np.exp(0.3*x) + 5

def df4(x):
    return -np.sin(2*x) - np.exp(0.3*x)*(0.5 + 0.15*x)

# MÉTODO DE BISECCIÓN CON ITERACIONES
def biseccion_con_iteraciones(f, a, b, tol=1e-6, max_iter=100):
    """Método de bisección mostrando iteraciones"""
    if f(a) * f(b) > 0:
        return None, []

    iteraciones = []
    for i in range(max_iter):
        c = (a + b) / 2
        fc = f(c)
        error = abs(b - a) / 2

        iteraciones.append({
            'iteracion': i + 1,
            'a': a,
            'b': b,
            'c': c,
            'f(c)': fc,
            'error': error
        })

        if abs(fc) < tol or error < tol:
            return c, iteraciones

        if f(a) * fc < 0:
            b = c
        else:
            a = c

    return (a + b) / 2, iteraciones

# MÉTODO DE NEWTON-RAPHSON CON ITERACIONES
def newton_con_iteraciones(f, df, x0, tol=1e-6, max_iter=100):
    """Método de Newton-Raphson mostrando iteraciones"""
    iteraciones = []
    x = x0

    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)

        if abs(dfx) < 1e-12:
            return None, iteraciones

        x_nuevo = x - fx / dfx
        error = abs(x_nuevo - x)

        iteraciones.append({
            'iteracion': i + 1,
            'x': x,
            'f(x)': fx,
            'f\'(x)': dfx,
            'x_nuevo': x_nuevo,
            'error': error
        })

        if abs(fx) < tol or error < tol:
            return x_nuevo, iteraciones

        x = x_nuevo

    return x, iteraciones

# MÉTODO DE LA SECANTE CON ITERACIONES
def secante_con_iteraciones(f, x0, x1, tol=1e-6, max_iter=100):
    """Método de la secante mostrando iteraciones"""
    iteraciones = []

    for i in range(max_iter):
        fx0 = f(x0)
        fx1 = f(x1)

        if abs(fx1 - fx0) < 1e-12:
            return None, iteraciones

        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        error = abs(x2 - x1)

        iteraciones.append({
            'iteracion': i + 1,
            'x0': x0,
            'x1': x1,
            'f(x0)': fx0,
            'f(x1)': fx1,
            'x2': x2,
            'error': error
        })

        if abs(fx1) < tol or error < tol:
            return x2, iteraciones

        x0, x1 = x1, x2

    return x1, iteraciones

# CONFIGURACIÓN DE LOS EJERCICIOS
ejercicios = {
    'EJERCICIO 1': {
        'funcion': f1,
        'derivada': df1,
        'intervalos': [(3, 4), (7, 8)],
        'puntos_newton': [3.5, 7.5],
        'rango_grafica': [-2, 10],
        'ecuacion': 'x³ - e^(0.8x) - 20 = 0'
    },
    'EJERCICIO 2': {
        'funcion': f2,
        'derivada': df2,
        'intervalos': [(4, 6), (8, 10)],
        'puntos_newton': [5.0, 9.0],
        'rango_grafica': [-5, 15],
        'ecuacion': '3·sin(0.5x) - 0.5x + 2 = 0'
    },
    'EJERCICIO 3': {
        'funcion': f3,
        'derivada': df3,
        'intervalos': [(-2, -1), (0, 1), (1.5, 2.5)],
        'puntos_newton': [-1.5, 0.5, 2.0],
        'rango_grafica': [-3, 3],
        'ecuacion': 'x³ - x²·e^(-0.5x) - 3x + 1 = 0'
    },
    'EJERCICIO 4': {
        'funcion': f4,
        'derivada': df4,
        'intervalos': [(3, 4), (6, 7)],
        'puntos_newton': [3.5, 6.5],
        'rango_grafica': [0, 10],
        'ecuacion': 'cos²(x) - 0.5x·e^(0.3x) + 5 = 0'
    }
}

# ANÁLISIS POR MÉTODO - BISECCIÓN
print("=" * 80)
print("MÉTODO DE BISECCIÓN - ANÁLISIS COMPLETO")
print("=" * 80)

for nombre, config in ejercicios.items():
    print(f"\n{nombre}")
    print(f"Función: {config['ecuacion']}")
    print("-" * 60)

    f = config['funcion']

    for i, (a, b) in enumerate(config['intervalos']):
        print(f"\nRaíz {i+1} - Intervalo [{a}, {b}]:")

        raiz, iteraciones = biseccion_con_iteraciones(f, a, b)

        if raiz is not None:
            print(f"Raíz encontrada: {raiz:.8f}")
            print(f"f({raiz:.6f}) = {f(raiz):.2e}")
            print(f"Iteraciones realizadas: {len(iteraciones)}")

            # Mostrar primeras y últimas iteraciones
            if len(iteraciones) > 0:
                print("\nPrimeras iteraciones:")
                primeras = iteraciones[:3]
                tabla = []
                for it in primeras:
                    tabla.append([it['iteracion'], f"{it['a']:.6f}", f"{it['b']:.6f}",
                                f"{it['c']:.6f}", f"{it['f(c)']:.2e}", f"{it['error']:.2e}"])
                print(tabulate(tabla, headers=['Iter', 'a', 'b', 'c', 'f(c)', 'Error'], tablefmt='grid'))

                if len(iteraciones) > 3:
                    print("\nÚltimas iteraciones:")
                    ultimas = iteraciones[-3:]
                    tabla = []
                    for it in ultimas:
                        tabla.append([it['iteracion'], f"{it['a']:.6f}", f"{it['b']:.6f}",
                                    f"{it['c']:.6f}", f"{it['f(c)']:.2e}", f"{it['error']:.2e}"])
                    print(tabulate(tabla, headers=['Iter', 'a', 'b', 'c', 'f(c)', 'Error'], tablefmt='grid'))
        else:
            print("No se encontró raíz en el intervalo (no hay cambio de signo)")

        print("-" * 40)

# GRÁFICAS PARA BISECCIÓN
fig, axs = plt.subplots(2, 2, figsize=(15, 12))
axs = axs.flatten()

for idx, (nombre, config) in enumerate(ejercicios.items()):
    f = config['funcion']
    x_range = config['rango_grafica']
    x = np.linspace(x_range[0], x_range[1], 1000)
    y = f(x)

    axs[idx].plot(x, y, 'b-', linewidth=2, label='f(x)')
    axs[idx].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axs[idx].grid(True, alpha=0.3)
    axs[idx].set_xlabel('x')
    axs[idx].set_ylabel('f(x)')
    axs[idx].set_title(f'{nombre}\n{config["ecuacion"]}')

    # Marcar intervalos de búsqueda
    for i, (a, b) in enumerate(config['intervalos']):
        axs[idx].axvspan(a, b, alpha=0.2, color='orange', label=f'Intervalo {i+1}' if i == 0 else "")

        # Encontrar y marcar raíz
        raiz, _ = biseccion_con_iteraciones(f, a, b)
        if raiz is not None:
            axs[idx].plot(raiz, f(raiz), 'ro', markersize=8,
                         label=f'Raíz {i+1}: {raiz:.4f}')

    axs[idx].legend()

plt.suptitle('MÉTODO DE BISECCIÓN - INTERVALOS Y RAÍCES', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# ANÁLISIS POR MÉTODO - NEWTON-RAPHSON
print("\n" + "=" * 80)
print("MÉTODO DE NEWTON-RAPHSON - ANÁLISIS COMPLETO")
print("=" * 80)

for nombre, config in ejercicios.items():
    print(f"\n{nombre}")
    print(f"Función: {config['ecuacion']}")
    print("-" * 60)

    f = config['funcion']
    df = config['derivada']

    for i, x0 in enumerate(config['puntos_newton']):
        print(f"\nRaíz {i+1} - Punto inicial: {x0}")

        raiz, iteraciones = newton_con_iteraciones(f, df, x0)

        if raiz is not None:
            print(f"Raíz encontrada: {raiz:.8f}")
            print(f"f({raiz:.6f}) = {f(raiz):.2e}")
            print(f"Iteraciones realizadas: {len(iteraciones)}")

            # Mostrar primeras y últimas iteraciones
            if len(iteraciones) > 0:
                print("\nPrimeras iteraciones:")
                primeras = iteraciones[:3]
                tabla = []
                for it in primeras:
                    tabla.append([it['iteracion'], f"{it['x']:.6f}", f"{it['f(x)']:.2e}",
                                f"{it['f\'(x)']:.2e}", f"{it['x_nuevo']:.6f}", f"{it['error']:.2e}"])
                print(tabulate(tabla, headers=['Iter', 'x', 'f(x)', 'f\'(x)', 'x_nuevo', 'Error'], tablefmt='grid'))

                if len(iteraciones) > 3:
                    print("\nÚltimas iteraciones:")
                    ultimas = iteraciones[-3:]
                    tabla = []
                    for it in ultimas:
                        tabla.append([it['iteracion'], f"{it['x']:.6f}", f"{it['f(x)']:.2e}",
                                    f"{it['f\'(x)']:.2e}", f"{it['x_nuevo']:.6f}", f"{it['error']:.2e}"])
                    print(tabulate(tabla, headers=['Iter', 'x', 'f(x)', 'f\'(x)', 'x_nuevo', 'Error'], tablefmt='grid'))
        else:
            print("El método no convergió (posible división por cero o derivada nula)")

        print("-" * 40)

# GRÁFICAS PARA NEWTON-RAPHSON
fig, axs = plt.subplots(2, 2, figsize=(15, 12))
axs = axs.flatten()

for idx, (nombre, config) in enumerate(ejercicios.items()):
    f = config['funcion']
    x_range = config['rango_grafica']
    x = np.linspace(x_range[0], x_range[1], 1000)
    y = f(x)

    axs[idx].plot(x, y, 'b-', linewidth=2, label='f(x)')
    axs[idx].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axs[idx].grid(True, alpha=0.3)
    axs[idx].set_xlabel('x')
    axs[idx].set_ylabel('f(x)')
    axs[idx].set_title(f'{nombre}\n{config["ecuacion"]}')

    # Marcar puntos iniciales y raíces encontradas
    for i, x0 in enumerate(config['puntos_newton']):
        axs[idx].plot(x0, f(x0), 'go', markersize=6,
                     label=f'Punto inicial {i+1}' if i == 0 else "")

        # Encontrar y marcar raíz
        raiz, _ = newton_con_iteraciones(f, config['derivada'], x0)
        if raiz is not None:
            axs[idx].plot(raiz, f(raiz), 'ro', markersize=8,
                         label=f'Raíz {i+1}: {raiz:.4f}')

    axs[idx].legend()

plt.suptitle('MÉTODO DE NEWTON-RAPHSON - PUNTOS INICIALES Y RAÍCES', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# ANÁLISIS POR MÉTODO - SECANTE
print("\n" + "=" * 80)
print("MÉTODO DE LA SECANTE - ANÁLISIS COMPLETO")
print("=" * 80)

for nombre, config in ejercicios.items():
    print(f"\n{nombre}")
    print(f"Función: {config['ecuacion']}")
    print("-" * 60)

    f = config['funcion']

    for i, (a, b) in enumerate(config['intervalos']):
        print(f"\nRaíz {i+1} - Puntos iniciales: {a}, {b}")

        raiz, iteraciones = secante_con_iteraciones(f, a, b)

        if raiz is not None:
            print(f"Raíz encontrada: {raiz:.8f}")
            print(f"f({raiz:.6f}) = {f(raiz):.2e}")
            print(f"Iteraciones realizadas: {len(iteraciones)}")

            # Mostrar primeras y últimas iteraciones
            if len(iteraciones) > 0:
                print("\nPrimeras iteraciones:")
                primeras = iteraciones[:3]
                tabla = []
                for it in primeras:
                    tabla.append([it['iteracion'], f"{it['x0']:.6f}", f"{it['x1']:.6f}",
                                f"{it['f(x0)']:.2e}", f"{it['f(x1)']:.2e}",
                                f"{it['x2']:.6f}", f"{it['error']:.2e}"])
                print(tabulate(tabla, headers=['Iter', 'x0', 'x1', 'f(x0)', 'f(x1)', 'x2', 'Error'], tablefmt='grid'))

                if len(iteraciones) > 3:
                    print("\nÚltimas iteraciones:")
                    ultimas = iteraciones[-3:]
                    tabla = []
                    for it in ultimas:
                        tabla.append([it['iteracion'], f"{it['x0']:.6f}", f"{it['x1']:.6f}",
                                    f"{it['f(x0)']:.2e}", f"{it['f(x1)']:.2e}",
                                    f"{it['x2']:.6f}", f"{it['error']:.2e}"])
                    print(tabulate(tabla, headers=['Iter', 'x0', 'x1', 'f(x0)', 'f(x1)', 'x2', 'Error'], tablefmt='grid'))
        else:
            print("El método no convergió (posible división por cero)")

        print("-" * 40)

# GRÁFICAS PARA SECANTE
fig, axs = plt.subplots(2, 2, figsize=(15, 12))
axs = axs.flatten()

for idx, (nombre, config) in enumerate(ejercicios.items()):
    f = config['funcion']
    x_range = config['rango_grafica']
    x = np.linspace(x_range[0], x_range[1], 1000)
    y = f(x)

    axs[idx].plot(x, y, 'b-', linewidth=2, label='f(x)')
    axs[idx].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axs[idx].grid(True, alpha=0.3)
    axs[idx].set_xlabel('x')
    axs[idx].set_ylabel('f(x)')
    axs[idx].set_title(f'{nombre}\n{config["ecuacion"]}')

    # Marcar puntos iniciales y raíces encontradas
    for i, (a, b) in enumerate(config['intervalos']):
        axs[idx].plot([a, b], [f(a), f(b)], 'go-', markersize=6,
                     label=f'Puntos iniciales {i+1}' if i == 0 else "")

        # Encontrar y marcar raíz
        raiz, _ = secante_con_iteraciones(f, a, b)
        if raiz is not None:
            axs[idx].plot(raiz, f(raiz), 'ro', markersize=8,
                         label=f'Raíz {i+1}: {raiz:.4f}')

    axs[idx].legend()

plt.suptitle('MÉTODO DE LA SECANTE - PUNTOS INICIALES Y RAÍCES', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# RESUMEN FINAL DE TODOS LOS MÉTODOS
print("\n" + "=" * 80)
print("RESUMEN COMPARATIVO - TODOS LOS MÉTODOS")
print("=" * 80)

for nombre, config in ejercicios.items():
    print(f"\n{nombre}: {config['ecuacion']}")
    print("-" * 60)

    f = config['funcion']
    df = config['derivada']

    resultados = []

    for i in range(len(config['intervalos'])):
        a, b = config['intervalos'][i]
        x0_newton = config['puntos_newton'][i]

        # Bisección
        raiz_bisec, iter_bisec = biseccion_con_iteraciones(f, a, b)
        # Newton
        raiz_newton, iter_newton = newton_con_iteraciones(f, df, x0_newton)
        # Secante
        raiz_sec, iter_sec = secante_con_iteraciones(f, a, b)

        resultados.append({
            'Raíz': i + 1,
            'Bisección': f"{raiz_bisec:.6f}" if raiz_bisec else "No converge",
            'Iter Bisec': len(iter_bisec) if raiz_bisec else "-",
            'Newton': f"{raiz_newton:.6f}" if raiz_newton else "No converge",
            'Iter Newton': len(iter_newton) if raiz_newton else "-",
            'Secante': f"{raiz_sec:.6f}" if raiz_sec else "No converge",
            'Iter Secante': len(iter_sec) if raiz_sec else "-"
        })

    # Mostrar tabla resumen
    tabla = []
    for res in resultados:
        tabla.append([res['Raíz'], res['Bisección'], res['Iter Bisec'],
                     res['Newton'], res['Iter Newton'],
                     res['Secante'], res['Iter Secante']])

    print(tabulate(tabla, headers=['Raíz', 'Bisección', 'Iter', 'Newton', 'Iter', 'Secante', 'Iter'],
                  tablefmt='grid'))

    print("\nConclusiones:")
    for res in resultados:
        if res['Bisección'] != "No converge" and res['Newton'] != "No converge" and res['Secante'] != "No converge":
            raiz_b = float(res['Bisección'])
            raiz_n = float(res['Newton'])
            raiz_s = float(res['Secante'])
            diff = max(abs(raiz_b - raiz_n), abs(raiz_b - raiz_s), abs(raiz_n - raiz_s))
            print(f"  Raíz {res['Raíz']}: Los tres métodos coinciden (diferencia máxima: {diff:.2e})")
        else:
            print(f"  Raíz {res['Raíz']}: Algunos métodos no convergieron")

print("\n" + "=" * 80)
print("ANÁLISIS FINAL COMPLETADO")
print("=" * 80)