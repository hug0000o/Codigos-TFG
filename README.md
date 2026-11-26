# Codigos-TFG

-prueba pbc.py: Script donde he diseñado y probado si funcionan correctamente las condiciones de contorno periódicas (PBC) 
para la cruz 2D. Como método extra de comprobación, se ha implementado la proyección 3D del sistema para
visualizar mejor su comportamiento.

-analisis_complejidad_temporal.py: este es simplmente para ver que el tiempo de la simulación escale linealmente con el número de partículas, y como lo hace. 

-MC traslacional.py: simplemente la pr

-MC completo (ani).py: script con toda la parte inicial de compresión ya terminada. Este de aquí saca una animación del proceso en la que se ve como los lados de las fronteras se van haciendo más pequeños por la compresión.

-MC completo (res).py: exactamente igual que el 'MC completo (ani).py' pero sin sacar animación. Simplemente enseña una imagen del estado final, indicando su fracción de empaquetamiento. 

-NB general.ipynb: recopilación con un poco más de explicación de todas las funciones que he hecho hasta ahora.

-modulo_tfg.py: modulo que actua como libreria del proyecto par aimportar las funciones desde aqui a los nuevos scripts que vaya haciendo

IMPORTANTE: al añadir 'modulo_tfg.py' solo he modificado el script 'MC completo (res)', por lo que los demas quedan un poco obsoletos y están aquí más por referencia, para ver la progresión histórica del trabajo. A partir de ahora el 'MC completo (res)' será el script de referencia para probar los parámetros y ajustar lo necesario. Obviamente tambien he actualizado el NB general cambiando la 'mc_step' por su nueva versión. Ya que es quí donde se explican realmente cada función y sus cosas particulares.
