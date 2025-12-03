import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import load_img, img_to_array

print("=" * 60)
print("LOADING TRAINED MODEL")
print("=" * 60)

# Load model architecture
with open('cifar10_model.json', 'r') as json_file:
    loaded_model_json = json_file.read()

model = model_from_json(loaded_model_json)
print("✓ Model architecture loaded from 'cifar10_model.json'")

# Load weights
model.load_weights('cifar10_model.weights.h5')
print("✓ Model weights loaded from 'cifar10_model.weights.h5'")

# Compile model (required for predictions)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print("✓ Model compiled and ready for predictions")

# Class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

classes_ua = ['літак', 'автомобіль', 'птах', 'кіт', 'олень',
              'собака', 'жаба', 'кінь', 'корабель', 'вантажівка']

print(f"\nClasses: {', '.join(class_names)}")

def predict_custom_image(img_path, show_probabilities=True):
    """
    Predict class of a custom image

    Parameters:
    - img_path: path to image file
    - show_probabilities: whether to show all class probabilities
    """
    print("\n" + "=" * 60)
    print(f"ANALYZING IMAGE: {img_path}")
    print("=" * 60)

    try:
        # Load and display original image
        img_original = load_img(img_path)

        # Load and resize to 32x32
        img_resized = load_img(img_path, target_size=(32, 32))

        # Display both images
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        ax1.imshow(img_original)
        ax1.set_title('Original Image', fontsize=12)
        ax1.axis('off')

        ax2.imshow(img_resized)
        ax2.set_title('Resized to 32x32', fontsize=12)
        ax2.axis('off')

        plt.tight_layout()
        output_name = img_path.replace('.jpg', '_comparison.png').replace('.png', '_comparison.png')
        plt.savefig(output_name, dpi=150, bbox_inches='tight')
        print(f"\n✓ Image comparison saved as '{output_name}'")
        plt.show()

        # Convert to array and preprocess
        x = img_to_array(img_resized)
        x = x / 255.0  # Normalize to [0, 1]
        x = np.expand_dims(x, axis=0)  # Add batch dimension

        print(f"\nProcessed image shape: {x.shape}")
        print(f"Pixel value range: [{x.min():.3f}, {x.max():.3f}]")

        # Make prediction
        print("\nMaking prediction...")
        prediction = model.predict(x, verbose=0)
        predicted_class_idx = np.argmax(prediction)
        confidence = prediction[0][predicted_class_idx]

        # Results
        predicted_class_en = class_names[predicted_class_idx]
        predicted_class_ua = classes_ua[predicted_class_idx]

        print("\n" + "=" * 60)
        print("PREDICTION RESULTS")
        print("=" * 60)
        print(f"Predicted Class: {predicted_class_en.upper()}")
        print(f"Ukrainian: {predicted_class_ua.upper()}")
        print(f"Confidence: {confidence:.4f} ({confidence * 100:.2f}%)")

        if show_probabilities:
            print(f"\n{'Class':<15} {'Probability':<12} {'Bar'}")
            print("-" * 60)

            # Sort by probability
            sorted_indices = np.argsort(prediction[0])[::-1]

            for idx in sorted_indices:
                prob = prediction[0][idx]
                bar = '█' * int(prob * 50)
                symbol = '★' if idx == predicted_class_idx else ' '
                print(f"{symbol} {class_names[idx]:<13} {prob:.4f} ({prob * 100:5.2f}%)  {bar}")

        # Visualization of prediction
        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.barh(class_names, prediction[0], color='skyblue')
        bars[predicted_class_idx].set_color('green')

        ax.set_xlabel('Probability', fontsize=12)
        ax.set_title(f'Class Probabilities\nPredicted: {predicted_class_en} ({confidence * 100:.1f}%)',
                     fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        output_name = img_path.replace('.jpg', '_probabilities.png').replace('.png', '_probabilities.png')
        plt.savefig(output_name, dpi=150, bbox_inches='tight')
        print(f"\n✓ Probability chart saved as '{output_name}'")
        plt.show()

        return predicted_class_en, predicted_class_ua, confidence

    except FileNotFoundError:
        print(f"\n❌ Error: File '{img_path}' not found!")
        print("Make sure the image file is in the same directory as this script.")
        return None, None, None
    except Exception as e:
        print(f"\n❌ Error processing image: {e}")
        return None, None, None

def predict_multiple_images(image_paths):
    """
    Predict classes for multiple images and display results

    Parameters:
    - image_paths: list of image file paths
    """
    results = []

    n_images = len(image_paths)
    fig, axes = plt.subplots((n_images + 2) // 3, 3, figsize=(15, 5 * ((n_images + 2) // 3)))
    if n_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, img_path in enumerate(image_paths):
        try:
            # Load and resize
            img = load_img(img_path, target_size=(32, 32))
            x = img_to_array(img) / 255.0
            x = np.expand_dims(x, axis=0)

            # Predict
            prediction = model.predict(x, verbose=0)
            predicted_class_idx = np.argmax(prediction)
            confidence = prediction[0][predicted_class_idx]

            # Store results
            results.append({
                'path': img_path,
                'class_en': class_names[predicted_class_idx],
                'class_ua': classes_ua[predicted_class_idx],
                'confidence': confidence
            })

            # Display
            axes[idx].imshow(img)
            axes[idx].set_title(f"{img_path}\n{class_names[predicted_class_idx]}\n{confidence * 100:.1f}%",
                                fontsize=10)
            axes[idx].axis('off')

        except Exception as e:
            print(f"Error with {img_path}: {e}")
            axes[idx].text(0.5, 0.5, f"Error loading\n{img_path}",
                           ha='center', va='center')
            axes[idx].axis('off')

    # Hide unused subplots
    for idx in range(len(image_paths), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig('batch_predictions.png', dpi=150, bbox_inches='tight')
    print("\n✓ Batch predictions saved as 'batch_predictions.png'")
    plt.show()

    return results

print("\n" + "=" * 60)
print("USAGE INSTRUCTIONS")
print("=" * 60)
print("\n1. Test single image:")
print("   predict_custom_image('dog3.jpg')")
print("\n2. Test multiple images:")
print("   image_list = ['dog3.jpg', 'cat1.jpg', 'airplane.jpg']")
print("   results = predict_multiple_images(image_list)")
print("\n3. Test without showing all probabilities:")
print("   predict_custom_image('image.jpg', show_probabilities=False)")

print("\n" + "="*60)
print("Testing with 'airplane.jpg'")
print("="*60)
predict_custom_image('airplane.jpg')
print("\n" + "="*60)
print("Testing with 'automobile.jpg'")
print("="*60)
predict_custom_image('automobile.jpg')
print("\n" + "="*60)
print("Testing with 'bird.jpg'")
print("="*60)
predict_custom_image('bird.jpg')
print("\n" + "="*60)
print("Testing with 'cat.jpg'")
print("="*60)
predict_custom_image('cat.jpg')
print("\n" + "="*60)
print("Testing with 'deer.jpg'")
print("="*60)
predict_custom_image('deer.jpg')
print("\n" + "="*60)
print("Testing with 'dog.jpg'")
print("="*60)
predict_custom_image('dog.jpg')
print("\n" + "="*60)
print("Testing with 'frog.jpg'")
print("="*60)
predict_custom_image('frog.jpg')
print("\n" + "="*60)
print("Testing with 'horse.jpg'")
print("="*60)
predict_custom_image('horse.jpg')
print("\n" + "="*60)
print("Testing with 'ship.jpg'")
print("="*60)
predict_custom_image('ship.jpg')
print("\n" + "="*60)
print("Testing with 'truck.jpg'")
print("="*60)
predict_custom_image('truck.jpg')
