import argparse
import cv2


def main():
    parser = argparse.ArgumentParser(description='Detecta rostos em uma imagem.')
    parser.add_argument('--image', required=True, help='Caminho da imagem de entrada')
    parser.add_argument('--output', default='output.jpg', help='Arquivo de saída com detecções')
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f'Não foi possível abrir {args.image}')
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imwrite(args.output, img)
    print(f'Detectado(s) {len(faces)} rosto(s). Resultado salvo em {args.output}')


if __name__ == '__main__':
    main()
