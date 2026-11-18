from django import forms

from .models import TIPOS_INFRACCION

TAMANO_MAXIMO = 5 * 1024 * 1024  # 5 MB


class AnalizarImagenForm(forms.Form):
    imagen = forms.FileField(
        label="Imagen",
        help_text="JPG o PNG, máximo 5 MB.",
        widget=forms.ClearableFileInput(attrs={"accept": "image/jpeg,image/png"}),
    )

    def clean_imagen(self):
        imagen = self.cleaned_data["imagen"]
        if imagen.size > TAMANO_MAXIMO:
            raise forms.ValidationError("La imagen supera los 5 MB.")
        return imagen


class FiltroIncidenciasForm(forms.Form):
    fecha = forms.DateField(required=False, widget=forms.DateInput(attrs={"type": "date"}))
    tipo = forms.ChoiceField(required=False, choices=[("", "Todos los tipos")] + TIPOS_INFRACCION)
