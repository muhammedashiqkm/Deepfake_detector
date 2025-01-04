from django.shortcuts import render, redirect
from django.views.generic import TemplateView, CreateView
from django.http import JsonResponse
from django.urls import reverse_lazy
from .models import Analysis
from .forms import ImageAnalysisForm, VideoAnalysisForm
from .utils import DeepfakeDetector
import json
import logging

logger = logging.getLogger(__name__)

# Initialize the detector
detector = DeepfakeDetector()

class HomeView(TemplateView):
    template_name = 'detector/home.html'

class ImageAnalysisView(CreateView):
    model = Analysis
    form_class = ImageAnalysisForm
    template_name = 'detector/image_analysis.html'
    success_url = reverse_lazy('image_result')
    
    def form_valid(self, form):
        # Save the analysis object
        analysis = form.save(commit=False)
        analysis.analysis_type = 'image'
        analysis.save()
        
        try:
            # Process the image
            results = detector.analyze_image(analysis.file.path)
            
            if results['error']:
                form.add_error(None, results['error'])
                analysis.delete()  # Clean up the analysis object
                return self.form_invalid(form)
            
            if not results['results']:
                form.add_error(None, "No valid analysis results")
                analysis.delete()  # Clean up the analysis object
                return self.form_invalid(form)
            
            # Update analysis with results
            best_confidence = max(r['confidence'] for r in results['results'])
            predictions = [r['prediction'] for r in results['results']]
            majority_prediction = max(set(predictions), key=predictions.count)
            
            analysis.result = majority_prediction
            analysis.confidence = best_confidence
            analysis.processed_file = results['visualization_path']
            analysis.save()
            
            # Store results in session for display
            self.request.session['analysis_results'] = {
                'id': analysis.id,
                'results': results['results'],
                'visualization_path': results['visualization_path']
            }
            
            return redirect('image_result')
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            form.add_error(None, str(e))
            analysis.delete()  # Clean up the analysis object
            return self.form_invalid(form)

class VideoAnalysisView(CreateView):
    model = Analysis
    form_class = VideoAnalysisForm
    template_name = 'detector/video_analysis.html'
    success_url = reverse_lazy('video_result')
    
    def form_valid(self, form):
        # Save the analysis object
        analysis = form.save(commit=False)
        analysis.analysis_type = 'video'
        analysis.save()
        
        try:
            # Process the video
            num_frames = form.cleaned_data.get('num_frames', 100)
            results = detector.analyze_video(analysis.file.path, num_frames)
            
            if results['error']:
                form.add_error(None, results['error'])
                analysis.delete()  # Clean up the analysis object
                return self.form_invalid(form)
            
            if not results['results']:
                form.add_error(None, "No valid analysis results")
                analysis.delete()  # Clean up the analysis object
                return self.form_invalid(form)
            
            # Update analysis with results
            best_confidence = max(r['confidence'] for r in results['results'])
            predictions = [r['prediction'] for r in results['results']]
            majority_prediction = max(set(predictions), key=predictions.count)
            
            analysis.result = majority_prediction
            analysis.confidence = best_confidence
            analysis.total_frames = results['total_frames_analyzed']
            analysis.fake_frame_ratio = max(r['fake_frame_ratio'] for r in results['results'])
            analysis.save()
            
            # Store results in session for display
            self.request.session['analysis_results'] = {
                'id': analysis.id,
                'results': results['results'],
                'frame_results': results['frame_results'],
                'total_frames': results['total_frames_analyzed'],
                'frame_images': results.get('frame_images', [])  
            }
            
            return redirect('video_result')
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            form.add_error(None, str(e))
            analysis.delete()  # Clean up the analysis object
            return self.form_invalid(form)

class ImageResultView(TemplateView):
    template_name = 'detector/image_result.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        results = self.request.session.get('analysis_results')
        
        if results:
            analysis = Analysis.objects.get(id=results['id'])
            context.update({
                'analysis': analysis,
                'results': results['results'],
                'visualization_path': results['visualization_path']
            })
            
        return context

class VideoResultView(TemplateView):
    template_name = 'detector/video_result.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        results = self.request.session.get('analysis_results')
        
        if results:
            analysis = Analysis.objects.get(id=results['id'])
            context.update({
                'analysis': analysis,
                'results': results['results'],
                'frame_results': results['frame_results'],
                'total_frames': results['total_frames'],
                'frame_images': results.get('frame_images', [])
            })
            
        return context
