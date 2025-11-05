"""
Zen VL extension of Agent Data Protocol (ADP) for multimodal visual observations.

This extends the original ADP schema from neulab/agent-data-protocol to support
visual observations (images, screenshots) alongside the standard text and web observations.

Original ADP: https://github.com/neulab/agent-data-protocol
Paper: https://arxiv.org/abs/2510.24702

Zen VL Extensions:
- ImageObservation: Screenshots, photos, diagrams
- VideoObservation: Video frames for temporal reasoning
- MultimodalAction: Actions that combine text and visual context
"""

from typing import Optional, Union, List, Dict, Any, Literal
from pydantic import BaseModel, Field
from enum import Enum


class ObservationSource(str, Enum):
    """Source of an observation"""
    USER = "user"
    ENVIRONMENT = "environment"
    SYSTEM = "system"


class ImageFormat(str, Enum):
    """Supported image formats"""
    PNG = "png"
    JPEG = "jpeg"
    JPG = "jpg"
    WEBP = "webp"
    BASE64 = "base64"


# ============================================================================
# ACTIONS (from original ADP + Zen VL extensions)
# ============================================================================

class APIAction(BaseModel):
    """
    API/Function call action with structured parameters.
    
    Examples:
    - goto(url="https://google.com")
    - click_element(x=120, y=200)
    - extract_text(image_path="screenshot.png", language="en")
    """
    type: Literal["api_action"] = "api_action"
    function: str = Field(description="Name of the function/API to call")
    kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Function arguments as key-value pairs"
    )
    description: Optional[str] = Field(
        default=None,
        description="Optional reasoning or explanation for this action"
    )


class CodeAction(BaseModel):
    """
    Code generation and execution action.
    
    Examples:
    - Python code: print("Hello World")
    - Bash command: ls -la
    - SQL query: SELECT * FROM users
    """
    type: Literal["code_action"] = "code_action"
    language: str = Field(description="Programming language (python, bash, sql, etc.)")
    content: str = Field(description="The code to execute")
    description: Optional[str] = Field(
        default=None,
        description="Optional reasoning or explanation for this code"
    )


class MessageAction(BaseModel):
    """
    Natural language communication action.
    
    Examples:
    - "I'll analyze this image for you"
    - "Let me click on the login button"
    - "Here's what I found in the screenshot"
    """
    type: Literal["message_action"] = "message_action"
    content: str = Field(description="The message content")
    role: Optional[str] = Field(
        default="assistant",
        description="Role of the message sender (assistant, user, system)"
    )


class MultimodalAction(BaseModel):
    """
    Zen VL Extension: Action that combines text reasoning with visual context.
    
    This enables the model to reason about visual content while taking actions.
    
    Example:
    - "I see a login form with username/password fields [image_ref: screenshot_1.png].
       I'll fill in the username field at coordinates (120, 200)."
    """
    type: Literal["multimodal_action"] = "multimodal_action"
    text: str = Field(description="Text reasoning or instruction")
    image_refs: List[str] = Field(
        default_factory=list,
        description="References to images being reasoned about"
    )
    action: Union[APIAction, CodeAction, MessageAction] = Field(
        description="The underlying action to take"
    )


Action = Union[APIAction, CodeAction, MessageAction, MultimodalAction]


# ============================================================================
# OBSERVATIONS (from original ADP + Zen VL extensions)
# ============================================================================

class TextObservation(BaseModel):
    """
    Text observation from user or environment.
    
    Examples:
    - User instruction: "Analyze this image"
    - Command output: "Hello World"
    - Error message: "File not found"
    """
    type: Literal["text_observation"] = "text_observation"
    source: ObservationSource = Field(description="Source of the observation")
    content: str = Field(description="The observed text")


class WebObservation(BaseModel):
    """
    Web page observation with HTML and accessibility tree.
    
    Examples:
    - Full webpage state after navigation
    - Form elements and their properties
    - Interactive UI components
    """
    type: Literal["web_observation"] = "web_observation"
    url: str = Field(description="Current page URL")
    html: Optional[str] = Field(default=None, description="Raw HTML content")
    axtree: Optional[str] = Field(
        default=None,
        description="Accessibility tree representation"
    )
    viewport_size: Optional[Dict[str, int]] = Field(
        default=None,
        description="Browser viewport dimensions {width, height}"
    )
    screenshot: Optional[str] = Field(
        default=None,
        description="Base64-encoded screenshot or path to screenshot file"
    )


class ImageObservation(BaseModel):
    """
    Zen VL Extension: Pure image observation for visual understanding.
    
    This is the core observation type for vision-language models, supporting:
    - Screenshots from GUI/web automation
    - Photos for visual analysis
    - Diagrams, charts, and infographics
    - Document images (receipts, forms, etc.)
    
    Examples:
    - Screenshot of application UI
    - Photo of a beach scene
    - Chart showing sales data
    - Invoice/receipt image
    """
    type: Literal["image_observation"] = "image_observation"
    source: ObservationSource = Field(description="Source of the image")
    image_path: Optional[str] = Field(
        default=None,
        description="Path to image file (relative or absolute)"
    )
    image_data: Optional[str] = Field(
        default=None,
        description="Base64-encoded image data"
    )
    image_url: Optional[str] = Field(
        default=None,
        description="URL to image (for remote images)"
    )
    format: ImageFormat = Field(
        default=ImageFormat.PNG,
        description="Image format"
    )
    width: Optional[int] = Field(default=None, description="Image width in pixels")
    height: Optional[int] = Field(default=None, description="Image height in pixels")
    caption: Optional[str] = Field(
        default=None,
        description="Optional caption or description"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata (timestamp, device, location, etc.)"
    )


class VideoObservation(BaseModel):
    """
    Zen VL Extension: Video observation for temporal visual understanding.
    
    Supports:
    - Screen recordings
    - Tutorial videos
    - Multi-frame temporal reasoning
    
    Examples:
    - Recording of GUI interaction sequence
    - Tutorial video showing workflow
    - Time-series visual data
    """
    type: Literal["video_observation"] = "video_observation"
    source: ObservationSource = Field(description="Source of the video")
    video_path: Optional[str] = Field(
        default=None,
        description="Path to video file"
    )
    video_url: Optional[str] = Field(
        default=None,
        description="URL to video"
    )
    frames: Optional[List[ImageObservation]] = Field(
        default_factory=list,
        description="Extracted key frames from video"
    )
    duration_seconds: Optional[float] = Field(
        default=None,
        description="Video duration in seconds"
    )
    fps: Optional[int] = Field(default=None, description="Frames per second")
    transcript: Optional[str] = Field(
        default=None,
        description="Optional audio transcript"
    )


Observation = Union[TextObservation, WebObservation, ImageObservation, VideoObservation]


# ============================================================================
# TRAJECTORY (from original ADP)
# ============================================================================

class Trajectory(BaseModel):
    """
    Complete agent trajectory as alternating sequence of actions and observations.
    
    Structure:
    - Starts with observation (user instruction or initial state)
    - Alternates between actions and observations
    - Ends with final observation or action
    
    Example:
    [
        TextObservation("Analyze this image"),
        ImageObservation(screenshot.png),
        MultimodalAction("I see a login form..."),
        APIAction(function="fill_form", kwargs={...}),
        TextObservation("Form filled successfully"),
        MessageAction("Task complete")
    ]
    """
    id: str = Field(description="Unique trajectory identifier")
    content: List[Union[Action, Observation]] = Field(
        description="Alternating sequence of actions and observations"
    )
    details: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Metadata (dataset source, task description, etc.)"
    )
    
    def validate_alternating(self) -> bool:
        """Validate that trajectory alternates between actions and observations."""
        if not self.content:
            return True
        
        is_action = lambda x: isinstance(x, (APIAction, CodeAction, MessageAction, MultimodalAction))
        is_observation = lambda x: isinstance(x, (TextObservation, WebObservation, ImageObservation, VideoObservation))
        
        for i in range(len(self.content) - 1):
            current_is_action = is_action(self.content[i])
            next_is_action = is_action(self.content[i + 1])
            
            # Should alternate: action -> observation or observation -> action
            if current_is_action == next_is_action:
                return False
        
        return True
    
    def get_actions(self) -> List[Action]:
        """Extract all actions from trajectory."""
        is_action = lambda x: isinstance(x, (APIAction, CodeAction, MessageAction, MultimodalAction))
        return [x for x in self.content if is_action(x)]
    
    def get_observations(self) -> List[Observation]:
        """Extract all observations from trajectory."""
        is_observation = lambda x: isinstance(x, (TextObservation, WebObservation, ImageObservation, VideoObservation))
        return [x for x in self.content if is_observation(x)]
    
    def get_visual_observations(self) -> List[Union[ImageObservation, VideoObservation]]:
        """Extract all visual observations (images and videos)."""
        return [x for x in self.content if isinstance(x, (ImageObservation, VideoObservation))]


# ============================================================================
# CONVERSION UTILITIES
# ============================================================================

def convert_adp_to_zen_vl_format(trajectory: Trajectory) -> Dict[str, Any]:
    """
    Convert ADP trajectory to Zen VL training format.
    
    This transforms:
    - Actions → Model outputs (with thinking chains)
    - Observations → Model inputs (multimodal)
    
    Returns a training example with:
    - images: List of image paths/data
    - messages: Conversation with system/user/assistant roles
    """
    messages = []
    images = []
    
    # System prompt
    messages.append({
        "role": "system",
        "content": "You are Zen VL, a vision-language model with function calling capabilities from Hanzo AI."
    })
    
    for item in trajectory.content:
        if isinstance(item, (TextObservation, WebObservation)):
            # User message
            if isinstance(item, TextObservation):
                content = item.content
            else:  # WebObservation
                content = f"URL: {item.url}\n"
                if item.axtree:
                    content += f"Accessibility Tree:\n{item.axtree[:500]}..."
                elif item.html:
                    content += f"HTML:\n{item.html[:500]}..."
            
            messages.append({
                "role": "user",
                "content": content
            })
            
        elif isinstance(item, ImageObservation):
            # Add image to list
            if item.image_path:
                images.append(item.image_path)
            elif item.image_data:
                images.append(item.image_data)
            
            # User message with image reference
            caption = item.caption or "Here is an image for analysis"
            messages.append({
                "role": "user",
                "content": f"[Image {len(images)}]: {caption}"
            })
            
        elif isinstance(item, (APIAction, CodeAction)):
            # Assistant action with thinking
            if isinstance(item, APIAction):
                thinking = item.description or "I'll call this function"
                function_call = {
                    "name": item.function,
                    "arguments": item.kwargs
                }
                response = {
                    "thinking": thinking,
                    "function_call": function_call
                }
            else:  # CodeAction
                thinking = item.description or "I'll execute this code"
                response = {
                    "thinking": thinking,
                    "code": {
                        "language": item.language,
                        "content": item.content
                    }
                }
            
            messages.append({
                "role": "assistant",
                "content": str(response)
            })
            
        elif isinstance(item, MessageAction):
            # Simple assistant message
            messages.append({
                "role": "assistant",
                "content": item.content
            })
    
    return {
        "id": trajectory.id,
        "images": images,
        "messages": messages,
        "metadata": trajectory.details
    }


# Example usage
if __name__ == "__main__":
    # Example: Image analysis trajectory
    trajectory = Trajectory(
        id="example_001",
        content=[
            TextObservation(
                source=ObservationSource.USER,
                content="Analyze this beach scene"
            ),
            ImageObservation(
                source=ObservationSource.USER,
                image_path="beach.jpg",
                caption="Beach scene with people and umbrellas"
            ),
            MultimodalAction(
                text="I see a beach scene with people, umbrellas, and ocean",
                image_refs=["beach.jpg"],
                action=APIAction(
                    function="image_analysis",
                    kwargs={
                        "objects": ["people", "umbrellas", "beach", "ocean"],
                        "scene": "outdoor_beach",
                        "colors": ["blue", "yellow", "beige"]
                    },
                    description="Analyzing the beach scene systematically"
                )
            ),
            TextObservation(
                source=ObservationSource.ENVIRONMENT,
                content="Analysis complete"
            ),
            MessageAction(
                content="I've identified the key elements in this beach scene."
            )
        ],
        details={
            "dataset": "zen-vl-visual-analysis",
            "task_type": "image_analysis"
        }
    )
    
    # Validate
    print(f"Trajectory valid: {trajectory.validate_alternating()}")
    print(f"Actions: {len(trajectory.get_actions())}")
    print(f"Observations: {len(trajectory.get_observations())}")
    print(f"Visual observations: {len(trajectory.get_visual_observations())}")
    
    # Convert to training format
    training_example = convert_adp_to_zen_vl_format(trajectory)
    print(f"\nTraining example:")
    print(f"Images: {training_example['images']}")
    print(f"Messages: {len(training_example['messages'])}")
