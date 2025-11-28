"""
Módulo para detecção de expressões faciais e emoções em vídeos.

Este módulo utiliza DeepFace e OpenCV para processar vídeos frame a frame,
detectando faces e identificando as emoções dominantes, gerando um vídeo
de saída com anotações visuais.
"""

import cv2
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from tqdm import tqdm

try:
    from deepface import DeepFace
except ImportError as e:
    raise ImportError(
        "DeepFace não está instalado. Instale com: pip install deepface"
    ) from e

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constantes
DEFAULT_COLOR = (0, 255, 0)  # Verde em BGR
DEFAULT_THICKNESS = 2
DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX
DEFAULT_FONT_SCALE = 0.6
LABEL_OFFSET_Y = 10
MIN_LABEL_Y = 20
VIDEO_CODEC = 'mp4v'
DETECTOR_BACKEND = 'opencv'
EMOTION_ACTIONS = ['emotion']


def validate_video_path(video_path: str) -> None:
    """
    Valida se o caminho do vídeo existe e é um arquivo válido.

    Args:
        video_path: Caminho para o arquivo de vídeo

    Raises:
        FileNotFoundError: Se o arquivo não existir
        ValueError: Se o caminho não for um arquivo válido
    """
    if not video_path:
        raise ValueError("O caminho do vídeo não pode ser vazio")

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de vídeo não encontrado: {video_path}")

    if not path.is_file():
        raise ValueError(f"O caminho não é um arquivo: {video_path}")


def validate_output_path(output_path: Optional[str]) -> None:
    """
    Valida e cria o diretório de saída se necessário.

    Args:
        output_path: Caminho para o arquivo de saída

    Raises:
        ValueError: Se o diretório pai não puder ser criado
    """
    if output_path:
        output_dir = Path(output_path).parent
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Diretório de saída preparado: {output_dir}")
        except OSError as e:
            raise ValueError(f"Não foi possível criar o diretório de saída: {e}") from e


def process_face_detection(
    frame: cv2.typing.MatLike,
    face_data: Dict[str, Any]
) -> None:
    """
    Processa uma face detectada, desenhando anotações no frame.

    Args:
        frame: Frame do vídeo onde as anotações serão desenhadas
        face_data: Dicionário contendo dados da face detectada
    """
    try:
        # Obtém a região da face
        face_region = face_data.get('region', {})
        x = face_region.get('x', 0)
        y = face_region.get('y', 0)
        w = face_region.get('w', 0)
        h = face_region.get('h', 0)

        # Valida dimensões da região
        if w <= 0 or h <= 0:
            logger.warning("Dimensões inválidas da região da face detectada")
            return

        # Obtém a emoção dominante
        dominant_emotion = face_data.get('dominant_emotion')
        emotion_dict = face_data.get('emotion', {})

        if not dominant_emotion or not emotion_dict:
            logger.warning("Dados de emoção incompletos para a face detectada")
            return

        emotion_confidence = emotion_dict.get(dominant_emotion, 0.0)

        # Desenha retângulo ao redor da face
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            DEFAULT_COLOR,
            DEFAULT_THICKNESS
        )

        # Adiciona o texto da emoção acima da face
        label = f"{dominant_emotion} ({emotion_confidence:.1f}%)"
        label_y = max(y - LABEL_OFFSET_Y, MIN_LABEL_Y)
        cv2.putText(
            frame,
            label,
            (x, label_y),
            DEFAULT_FONT,
            DEFAULT_FONT_SCALE,
            DEFAULT_COLOR,
            DEFAULT_THICKNESS
        )
    except KeyError as e:
        logger.warning(f"Chave ausente nos dados da face: {e}")
    except Exception as e:
        logger.error(f"Erro ao processar face detectada: {e}", exc_info=True)


def get_video_properties(cap: cv2.VideoCapture) -> Optional[Dict[str, Any]]:
    """
    Obtém as propriedades do vídeo.

    Args:
        cap: Objeto VideoCapture aberto

    Returns:
        Dicionário com propriedades do vídeo ou None se inválido
    """
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frame_count <= 0:
        logger.warning("Número de frames inválido ou vídeo vazio")
        return None

    logger.info(
        f"Propriedades do vídeo: {frame_count} frames, "
        f"{fps:.2f} FPS, {width}x{height}"
    )

    return {
        'frame_count': frame_count,
        'fps': fps,
        'width': width,
        'height': height
    }


def setup_video_writer(
    output_path: str,
    fps: float,
    width: int,
    height: int
) -> Optional[cv2.VideoWriter]:
    """
    Configura o writer de vídeo para salvar o resultado.

    Args:
        output_path: Caminho para o arquivo de saída
        fps: Frames por segundo
        width: Largura do vídeo
        height: Altura do vídeo

    Returns:
        Objeto VideoWriter ou None se falhar
    """
    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        logger.error(f"Não foi possível criar o arquivo de saída: {output_path}")
        return None

    return out


def analyze_frame(frame: cv2.typing.MatLike) -> List[Dict[str, Any]]:
    """
    Analisa um frame detectando faces e emoções.

    Args:
        frame: Frame do vídeo a ser analisado

    Returns:
        Lista de dicionários com dados das faces detectadas
    """
    try:
        results = DeepFace.analyze(
            frame,
            actions=EMOTION_ACTIONS,
            enforce_detection=False,
            detector_backend=DETECTOR_BACKEND,
            silent=True
        )

        # Garante que results seja sempre uma lista
        if not isinstance(results, list):
            results = [results]

        return results
    except Exception as e:
        logger.warning(f"Erro ao analisar frame: {e}")
        return []


def process_single_frame(
    frame: cv2.typing.MatLike,
    display: bool,
    out: Optional[cv2.VideoWriter]
) -> int:
    """
    Processa um único frame do vídeo.

    Args:
        frame: Frame a ser processado
        display: Se True, exibe o frame
        out: Writer de vídeo para salvar o frame

    Returns:
        Número de faces detectadas no frame (-1 se interrompido pelo usuário)
    """
    faces_detected = 0
    results = analyze_frame(frame)

    for face_data in results:
        process_face_detection(frame, face_data)
        faces_detected += 1

    if display:
        cv2.imshow('Video - Pressione Q para sair', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info("Processamento interrompido pelo usuário")
            return -1  # Indica interrupção

    if out:
        out.write(frame)

    return faces_detected


def detect_expressions_in_video(
    video_path: str,
    output_path: Optional[str] = None,
    display: bool = False
) -> bool:
    """
    Detecta expressões faciais e emoções em um vídeo.

    Processa o vídeo frame a frame, detectando faces e identificando emoções,
    gerando um vídeo de saída com anotações visuais opcionais.

    Args:
        video_path: Caminho para o arquivo de vídeo de entrada
        output_path: Caminho opcional para salvar o vídeo processado.
                    Se None, o vídeo não será salvo
        display: Se True, exibe o vídeo em tempo real durante o processamento

    Returns:
        True se o processamento foi concluído com sucesso, False caso contrário

    Raises:
        FileNotFoundError: Se o arquivo de vídeo não existir
        ValueError: Se os parâmetros forem inválidos
    """
    # Validação de entrada
    validate_video_path(video_path)
    validate_output_path(output_path)

    logger.info(f"Iniciando processamento do vídeo: {video_path}")

    # Abre o vídeo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        error_msg = f"Não foi possível abrir o vídeo: {video_path}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    out = None
    try:
        # Obtém propriedades do vídeo
        props = get_video_properties(cap)
        if not props:
            return False

        # Configura o writer de vídeo se necessário
        if output_path:
            out = setup_video_writer(
                output_path,
                props['fps'],
                props['width'],
                props['height']
            )
            if not out:
                return False

        # Processa cada frame
        frames_processed = 0
        faces_detected = 0

        with tqdm(total=props['frame_count'], desc="Processando frames do vídeo") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                faces_in_frame = process_single_frame(
                    frame,
                    display,
                    out
                )

                if faces_in_frame == -1:  # Interrupção pelo usuário
                    break

                faces_detected += faces_in_frame
                frames_processed += 1
                pbar.update(1)

        logger.info(
            f"Processamento concluído: {frames_processed} frames processados, "
            f"{faces_detected} faces detectadas"
        )

        if output_path:
            logger.info(f"Vídeo de saída salvo em: {output_path}")

        return True

    finally:
        # Libera recursos
        cap.release()
        if out:
            out.release()
        if display:
            cv2.destroyAllWindows()
        logger.info("Recursos liberados")


def main() -> None:
    """Função principal para execução do script."""
    try:
        script_dir = Path(__file__).parent.resolve()
        video_file = script_dir / "videos" / "input_video.mp4"
        output_file = script_dir / "videos" / "output_video.mp4"

        success = detect_expressions_in_video(
            str(video_file),
            str(output_file),
            display=False
        )

        if success:
            logger.info("Processamento concluído com sucesso!")
        else:
            logger.error("Processamento falhou")

    except Exception as e:
        logger.error(f"Erro durante a execução: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()