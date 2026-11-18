// Panel en vivo: consulta /api/estado/ cada segundo y actualiza la interfaz.
(function () {
  const $ = (id) => document.getElementById(id);
  const csrf = document.querySelector("[name=csrfmiddlewaretoken]").value;

  function escapar(texto) {
    const div = document.createElement("div");
    div.textContent = texto;
    return div.innerHTML;
  }

  function pintar(e) {
    const cam = $("estado-camara");
    cam.textContent = e.camara ? "● En vivo" : "Cámara no disponible";
    cam.className = "pastilla " + (e.camara ? "ok" : "mal");

    $("m-personas").textContent = e.personas;
    $("m-con-mascarilla").textContent = e.mascarilla.con;
    $("m-sin-mascarilla").textContent = e.mascarilla.sin;
    $("m-incidencias").textContent = e.hoy.total;
    $("m-fps").textContent = `· ${e.fps.toFixed(0)} FPS`;
    $("motores").textContent =
      `Objetos: ${e.yolo ? "YOLOv4-tiny" : "sin modelo"} · Rostros: ${e.motor_rostros}` +
      (e.iluminacion_mejorada ? " · Luz baja: CLAHE activo" : "");

    const banner = $("banner-alerta");
    if (e.alertas.length) {
      banner.textContent = "⚠ " + e.alertas.map((a) => a.nombre).join(" · ");
      banner.classList.remove("oculto");
    } else {
      banner.classList.add("oculto");
    }

    const objetos = Object.entries(e.objetos);
    $("objetos").innerHTML = objetos.length
      ? objetos.map(([nombre, n]) =>
          `<span class="chip ${nombre === "Celular" ? "rojo" : ""}">${escapar(nombre)} × ${n}</span>`).join("")
      : '<span class="tenue">Ninguno</span>';

    $("eventos").innerHTML = e.eventos
      .map((ev) => `<li class="${ev.tipo}"><small>${ev.hora}</small> ${escapar(ev.texto)}</li>`).join("");

    $("incidencias").innerHTML = e.hoy.ultimas.length
      ? e.hoy.ultimas.map((i) => `
          <li>${i.captura ? `<img src="${i.captura}" alt="">` : ""}
            <div><strong>${escapar(i.tipo)}</strong><small>${i.hora} · ${i.confianza.toFixed(0)}%</small></div>
          </li>`).join("")
      : '<li class="tenue">Sin incidencias hoy</li>';
  }

  async function actualizar() {
    try {
      const r = await fetch(window.URL_ESTADO, { cache: "no-store" });
      if (r.ok) pintar(await r.json());
    } catch (err) {
      $("estado-camara").textContent = "Servidor sin respuesta";
      $("estado-camara").className = "pastilla mal";
    } finally {
      setTimeout(actualizar, 1000);
    }
  }

  document.querySelectorAll("[data-regla]").forEach((casilla) => {
    casilla.addEventListener("change", async () => {
      const url = window.URL_REGLA.replace("CLAVE", casilla.dataset.regla);
      try {
        const r = await fetch(url, { method: "POST", headers: { "X-CSRFToken": csrf } });
        const datos = await r.json();
        casilla.checked = datos.activa;
      } catch (err) {
        casilla.checked = !casilla.checked; // revertir si falló
      }
    });
  });

  // Si el stream MJPEG se corta (servidor reiniciado), reconectar.
  const video = $("video");
  video.addEventListener("error", () => setTimeout(() => { video.src = video.src.split("?")[0] + "?t=" + Date.now(); }, 2000));

  actualizar();
})();
