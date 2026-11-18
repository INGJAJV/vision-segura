"""Módulo de visión por computador (OpenCV).

- procesamiento:     operaciones de imagen (iluminación, segmentación de color).
- detector_objetos:  YOLOv4-tiny (cv2.dnn) sobre las 80 clases COCO.
- detector_rostros:  YuNet (cv2.FaceDetectorYN) con respaldo Haar Cascade.
- clasificador_epp:  decide con/sin mascarilla y con/sin casco.
- analizador:        une los detectores y dibuja los resultados.
- camara:            captura de video con reconexión automática.
- servicio:          pipeline en tiempo real + reglas e incidencias.

Solo `servicio.obtener_servicio` depende de Django; el resto es OpenCV puro
y se prueba de forma aislada.
"""
