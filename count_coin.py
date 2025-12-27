from ultralytics import YOLO
import cv2

# Charger le modèle
model = YOLO('runs/detect/cpoin_model/weights/best.pt')
cap = cv2.VideoCapture('coin_vd_1.mp4')

# Dimensions d'affichage
scale_percent = 50
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
display_width = int(original_width * scale_percent / 100)
display_height = int(original_height * scale_percent / 100)

max_coins = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Détection
    results = model.predict(
        frame,
        conf=0.25,
        iou=0.3,
        imgsz=1280,
        verbose=False
    )

    # Récupérer les boxes
    boxes = results[0].boxes
    num_coins = len(boxes)

    # Dessiner manuellement les bounding boxes
    for box in boxes:
        # Coordonnées (x1, y1, x2, y2)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Calcul largeur et hauteur
        width = x2 - x1
        height = y2 - y1

        # Centre
        center_x = x1 + width // 2
        center_y = y1 + height // 2

        # Confiance
        conf = float(box.conf[0])

        # Dessiner le rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Afficher les infos
        label = f"Piece {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Point central
        cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)

        # Afficher coordonnées (optionnel)
        coords_text = f"x:{center_x} y:{center_y}"
        cv2.putText(frame, coords_text, (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    # Mettre à jour le maximum
    if num_coins > max_coins:
        max_coins = num_coins
        print(f"Nouveau maximum: {max_coins} pièces")

    # Panneau d'information
    cv2.rectangle(frame, (10, 10), (300, 100), (0, 0, 0), -1)
    cv2.putText(frame, f"Pieces: {num_coins}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Max: {max_coins}", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Redimensionner et afficher
    resized = cv2.resize(frame, (display_width, display_height))
    cv2.imshow('Comptage de Pieces', resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Maximum detecte: {max_coins} pieces")
