import base64
import csv

from django.db.models import Count
from django.db.models.functions import ExtractHour
from django.http import HttpResponse, JsonResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from .forms import AnalizarImagenForm, FiltroIncidenciasForm
from .models import TIPOS_INFRACCION, Incidencia, Regla
from .vision import procesamiento as proc
from .vision.servicio import obtener_analizador, obtener_servicio


def _incidencias_de_hoy():
    return Incidencia.objects.filter(fecha_hora__date=timezone.localdate())


def _filtrar(request):
    form = FiltroIncidenciasForm(request.GET or None)
    fecha, tipo = timezone.localdate(), ""
    if form.is_valid():
        fecha = form.cleaned_data["fecha"] or fecha
        tipo = form.cleaned_data["tipo"]
    qs = Incidencia.objects.filter(fecha_hora__date=fecha)
    if tipo:
        qs = qs.filter(tipo=tipo)
    return form, fecha, tipo, qs


# ---------------------------------------------------------------- páginas
def panel(request):
    return render(request, "monitor/panel.html", {
        "reglas": Regla.objects.all(),
        "incidencias": _incidencias_de_hoy()[:8],
    })


def reportes(request):
    form, fecha, tipo, qs = _filtrar(request)
    nombres = dict(TIPOS_INFRACCION)
    por_tipo = [
        {"tipo": nombres.get(f["tipo"], f["tipo"]), "total": f["total"]}
        for f in qs.values("tipo").annotate(total=Count("id")).order_by("-total")
    ]
    por_hora = {
        f["hora"]: f["total"]
        for f in qs.annotate(hora=ExtractHour("fecha_hora")).values("hora").annotate(total=Count("id"))
    }
    maximo = max(por_hora.values(), default=0)
    horas = [
        {"hora": h, "total": por_hora.get(h, 0), "alto": round(100 * por_hora.get(h, 0) / maximo) if maximo else 0}
        for h in range(7, 22)
    ]
    return render(request, "monitor/reportes.html", {
        "form": form,
        "fecha": fecha,
        "tipo": tipo,
        "incidencias": qs,
        "total": qs.count(),
        "pendientes": qs.filter(revisada=False).count(),
        "por_tipo": por_tipo,
        "horas": horas,
    })


def exportar_csv(request):
    _, fecha, _, qs = _filtrar(request)
    respuesta = HttpResponse(content_type="text/csv; charset=utf-8")
    respuesta["Content-Disposition"] = f'attachment; filename="incidencias_{fecha}.csv"'
    respuesta.write("﻿")  # BOM: Excel muestra bien las tildes
    escritor = csv.writer(respuesta)
    escritor.writerow(["ID", "Tipo", "Fecha", "Hora", "Confianza (%)", "Personas", "Revisada", "Captura"])
    for i in qs:
        local = timezone.localtime(i.fecha_hora)
        escritor.writerow([
            i.id, i.get_tipo_display(), local.date(), local.strftime("%H:%M:%S"),
            i.confianza, i.personas, "Sí" if i.revisada else "No",
            request.build_absolute_uri(i.captura.url) if i.captura else "",
        ])
    return respuesta


@require_POST
def incidencia_revisar(request, pk):
    incidencia = get_object_or_404(Incidencia, pk=pk)
    incidencia.revisada = not incidencia.revisada
    incidencia.save(update_fields=["revisada"])
    return redirect(request.POST.get("siguiente") or "reportes")


def analizar(request):
    """Analiza una imagen subida con el mismo motor que el video en vivo."""
    form = AnalizarImagenForm(request.POST or None, request.FILES or None)
    contexto = {"form": form}
    if request.method == "POST" and form.is_valid():
        try:
            imagen = proc.decodificar_imagen(form.cleaned_data["imagen"].read())
        except ValueError as exc:
            form.add_error("imagen", str(exc))
        else:
            imagen = proc.redimensionar_max(imagen, 960)
            analizador = obtener_analizador()
            resultado = analizador.analizar(imagen, evaluar_casco=True)
            analizador.dibujar(imagen, resultado)
            contexto.update({
                "resultado": resultado.a_dict(),
                "infracciones": [
                    {"nombre": dict(TIPOS_INFRACCION)[k], "confianza": v}
                    for k, v in resultado.infracciones().items()
                ],
                "imagen_b64": base64.b64encode(proc.codificar_jpeg(imagen, 90)).decode(),
            })
    return render(request, "monitor/analizar.html", contexto)


# ------------------------------------------------------------ video y API
def video(request):
    servicio = obtener_servicio()

    def flujo():
        while True:
            jpeg = servicio.esperar_jpeg()
            if jpeg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

    return StreamingHttpResponse(flujo(), content_type="multipart/x-mixed-replace; boundary=frame")


def api_estado(request):
    estado = obtener_servicio().estado()
    hoy = _incidencias_de_hoy()
    estado["hoy"] = {
        "total": hoy.count(),
        "por_tipo": {f["tipo"]: f["total"] for f in hoy.values("tipo").annotate(total=Count("id"))},
        "ultimas": [
            {
                "tipo": i.get_tipo_display(),
                "hora": timezone.localtime(i.fecha_hora).strftime("%H:%M:%S"),
                "confianza": i.confianza,
                "captura": i.captura.url if i.captura else None,
            }
            for i in hoy[:8]
        ],
    }
    return JsonResponse(estado)


@require_POST
def api_regla(request, clave):
    regla = get_object_or_404(Regla, clave=clave)
    regla.activa = not regla.activa
    regla.save(update_fields=["activa"])
    obtener_servicio().recargar_reglas()
    return JsonResponse({"clave": regla.clave, "activa": regla.activa})
