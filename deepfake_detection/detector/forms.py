from django import forms
from .models import Analysis

class ImageAnalysisForm(forms.ModelForm):
    class Meta:
        model = Analysis
        fields = ['file']
        widgets = {
            'file': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'image/*',
                'data-browse-label': 'Choose image...',
            })
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['file'].label = "Choose Image"
        
    def clean_file(self):
        file = self.cleaned_data.get('file')
        if file:
            if not file.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                raise forms.ValidationError('Only PNG and JPG files are allowed.')
            if file.size > 10 * 1024 * 1024:  # 10MB limit
                raise forms.ValidationError('File size must be under 10MB.')
        return file

class VideoAnalysisForm(forms.ModelForm):
    num_frames = forms.IntegerField(
        min_value=10,
        max_value=300,
        initial=100,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'type': 'range',
            'step': '10'
        })
    )

    class Meta:
        model = Analysis
        fields = ['file']
        widgets = {
            'file': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'video/*',
                'data-browse-label': 'Choose video...',
            })
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['file'].label = "Choose Video"
        
    def clean_file(self):
        file = self.cleaned_data.get('file')
        if file:
            if not file.name.lower().endswith(('.mp4', '.avi', '.mov')):
                raise forms.ValidationError('Only MP4, AVI, and MOV files are allowed.')
            if file.size > 100 * 1024 * 1024:  # 100MB limit
                raise forms.ValidationError('File size must be under 100MB.')
        return file
