from PIL import Image, ImageOps
import os

class Sizing:
    def __init__(self, base_path='./train'):
        self.base_path = base_path

    def resize_and_save (self , image_path):
        with Image.open(image_path) as img:

            # Check if the image needs downsizing
            if img.width > 1920 or img.height > 1080:
                aspect_ratio = img.width / img.height
                if aspect_ratio > 1920 / 1080:  # Image is wider than desired aspect ratio
                    img = img.resize((1920 , int(1920 / aspect_ratio)))
                else:  # Image is taller than desired aspect ratio
                    img = img.resize((int(1080 * aspect_ratio) , 1080))

            delta_w = max(0 , 1920 - img.width)
            delta_h = max(0 , 1080 - img.height)
            padding = (delta_w // 2 , delta_h // 2 , delta_w - (delta_w // 2) , delta_h - (delta_h // 2))
            new_img = ImageOps.expand(img , padding , fill = "black")
            new_img = new_img.resize((1920 , 1080))  # This ensures that the image is 1920x1080
            new_img.save(image_path)

    def analyze_aspect_ratios(self, resize=False):
        all_aspect_ratios = []
        label_stats = {}
        resized_count = 0  # Counter for resized images
        tolerance = 0.15

        class_labels = [d for d in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, d))]
        for label in class_labels:
            label_aspect_ratios = []
            label_path = os.path.join(self.base_path, label)

            for image_name in os.listdir(label_path):
                if image_name.endswith('.jpeg'):
                    image_path = os.path.join(label_path, image_name)
                    with Image.open(image_path) as img:
                        width, height = img.size
                        aspect_ratio = width / height
                        label_aspect_ratios.append(aspect_ratio)
                        all_aspect_ratios.append(aspect_ratio)

                        # Check if the image needs resizing
                        if resize and abs(aspect_ratio - 1920/1080) > 0.2:
                            self.resize_and_save(image_path)
                            resized_count += 1

            mean_aspect_ratio_label = sum(label_aspect_ratios) / len(label_aspect_ratios)

            outliers_label = [aspect_ratio for aspect_ratio in label_aspect_ratios if abs(aspect_ratio - mean_aspect_ratio_label) > tolerance]

            label_stats[label] = {
                "mean_aspect_ratio": mean_aspect_ratio_label,
                "num_outliers": len(outliers_label),
                "percent_outliers": len(outliers_label) / len(label_aspect_ratios) * 100
            }

        mean_aspect_ratio = sum(all_aspect_ratios) / len(all_aspect_ratios)
        overall_outliers = [aspect_ratio for aspect_ratio in all_aspect_ratios if abs(aspect_ratio - mean_aspect_ratio) > tolerance]

        for label, stats in label_stats.items():
            print(f"Class Label: {label}")
            print(f"Mean Aspect Ratio: {stats['mean_aspect_ratio']:.4f}")
            print(f"Number of Outliers: {stats['num_outliers']}")
            print(f"Percentage of Outliers: {stats['percent_outliers']:.2f}%")
            print("-" * 20)

        print(f"Overall Mean Aspect Ratio: {mean_aspect_ratio:.4f}")
        print(f"Overall Number of Outliers: {len(overall_outliers)}")
        print(f"Overall Percentage of Outliers: {len(overall_outliers) / len(all_aspect_ratios) * 100:.2f}%")
        if resize:
            print(f"Resized {resized_count} images.")

        return mean_aspect_ratio, overall_outliers

if __name__ == "__main__":
    sizing = Sizing(base_path = "train")
    mean_aspect_ratio, outliers = sizing.analyze_aspect_ratios(resize=True)
