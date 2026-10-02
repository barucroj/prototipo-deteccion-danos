from setuptools import setup, find_packages

setup(
    name="prototipo-deteccion-danos",
    version="0.1.0",
    author="Antonio Baruc Rojas Cuapio",
    description="Prototipo de detección de daños vehiculares nuevos entre fotos antes/después "
                "de una renta (TT 27-1-0005, ESCOM-IPN)",
    packages=find_packages(),

    install_requires=[
    ],
    python_requires=">=3.9",
)