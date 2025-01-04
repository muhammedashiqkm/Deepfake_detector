from django.db import models
from django.utils import timezone

# Create your models here.

class Analysis(models.Model):
    ANALYSIS_TYPES = (
        ('image', 'Image'),
        ('video', 'Video'),
    )
    
    RESULT_TYPES = (
        ('real', 'Real'),
        ('fake', 'Fake'),
    )
    
    file = models.FileField(upload_to='uploads/')
    analysis_type = models.CharField(max_length=10, choices=ANALYSIS_TYPES)
    result = models.CharField(max_length=10, choices=RESULT_TYPES, null=True, blank=True)
    confidence = models.FloatField(null=True, blank=True)
    processed_file = models.FileField(upload_to='processed/', null=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now)
    
    # For video analysis
    total_frames = models.IntegerField(null=True, blank=True)
    fake_frame_ratio = models.FloatField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.analysis_type} Analysis - {self.created_at}"
