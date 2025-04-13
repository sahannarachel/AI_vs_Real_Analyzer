from PIL import Image, ExifTags

# NEW FEATURE 2: Metadata Analysis function
def analyze_metadata(image_path):
    """
    Extract and analyze metadata from image to detect AI generation signs
    """
    metadata_indicators = {
        'suspicious': False,
        'ai_signs': [],
        'human_signs': [],
        'raw_metadata': {}
    }
    
    try:
        # Open image and extract EXIF data
        with Image.open(image_path) as img:
            # Check if image has EXIF data
            if hasattr(img, '_getexif') and img._getexif() is not None:
                exif_data = {
                    ExifTags.TAGS.get(tag, tag): value
                    for tag, value in img._getexif().items()
                    if tag in ExifTags.TAGS
                }
                
                # Store raw metadata
                metadata_indicators['raw_metadata'] = exif_data
                
                # Check for common camera indicators
                if 'Make' in exif_data or 'Model' in exif_data:
                    metadata_indicators['human_signs'].append(f"Camera information found: {exif_data.get('Make', '')} {exif_data.get('Model', '')}")
                
                # Check for editing software signatures
                if 'Software' in exif_data:
                    software = exif_data['Software']
                    metadata_indicators['raw_metadata']['Software'] = software
                    
                    # Check for AI generation tools
                    ai_tools = ['stable diffusion', 'dall-e', 'midjourney', 'generative', 'ai', 'gan', 'neural', 'diffusion']
                    if any(tool.lower() in software.lower() for tool in ai_tools):
                        metadata_indicators['suspicious'] = True
                        metadata_indicators['ai_signs'].append(f"AI generation software detected: {software}")
                    else:
                        metadata_indicators['human_signs'].append(f"Standard editing software: {software}")
                
                # Check for unusual or missing EXIF fields common in AI images
                if 'DateTimeOriginal' not in exif_data and 'DateTime' not in exif_data:
                    metadata_indicators['ai_signs'].append("Missing timestamp information")
                    
                if 'ExifImageWidth' not in exif_data or 'ExifImageHeight' not in exif_data:
                    metadata_indicators['ai_signs'].append("Missing original dimension information")
            else:
                # No EXIF data is sometimes indicative of AI-generated images
                metadata_indicators['ai_signs'].append("No EXIF metadata found")
                
            # Check file format peculiarities
            if img.format == 'PNG' and not metadata_indicators['raw_metadata']:
                metadata_indicators['ai_signs'].append("Clean PNG with no metadata (common in AI outputs)")
                
    except Exception as e:
        print(f"Metadata analysis error: {e}")
        metadata_indicators['ai_signs'].append(f"Error reading metadata: {str(e)}")
    
    # Calculate a simple metadata-based score (basic implementation)
    ai_score = len(metadata_indicators['ai_signs']) 
    human_score = len(metadata_indicators['human_signs'])
    
    metadata_indicators['metadata_score'] = {
        'ai_likelihood': ai_score / (ai_score + human_score + 0.1),  # Avoid division by zero
        'human_likelihood': human_score / (ai_score + human_score + 0.1)
    }
    
    return metadata_indicators