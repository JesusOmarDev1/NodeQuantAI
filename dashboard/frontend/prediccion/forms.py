from django import forms

class PredicionForm(forms.Form):
    imagen = forms.FileField(label="Imagen (.nii.gz)")
    mascara = forms.FileField(label="Máscara (.nii.gz)")