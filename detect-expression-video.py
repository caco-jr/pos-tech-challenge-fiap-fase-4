"""
Módulo para detecção de expressões faciais e emoções em vídeos.

Este módulo utiliza DeepFace e OpenCV para processar vídeos frame a frame,
detectando faces e identificando as emoções dominantes, gerando um vídeo
de saída com anotações visuais.
"""

import cv2
import logging
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from collections import defaultdict, deque
from datetime import datetime
from tqdm import tqdm
import numpy as np

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
ANOMALY_COLOR = (0, 0, 255)  # Vermelho para anomalias
ACTIVITY_COLOR = (255, 165, 0)  # Laranja para atividades
DEFAULT_THICKNESS = 2
DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX
DEFAULT_FONT_SCALE = 0.6
LABEL_OFFSET_Y = 10
MIN_LABEL_Y = 20
VIDEO_CODEC = 'mp4v'
# Backends disponíveis: 'opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe'
# RetinaFace e MTCNN são mais precisos para detectar apenas humanos reais
DETECTOR_BACKEND = 'retinaface'  # Mais robusto para filtrar bonecos/ilustrações
EMOTION_ACTIONS = ['emotion']

# Constantes para detecção de atividades e anomalias
MOVEMENT_THRESHOLD = 30  # Pixels de movimento para considerar atividade
ANOMALY_MOVEMENT_THRESHOLD = 80  # Pixels de movimento brusco para anomalia
EMOTION_CHANGE_THRESHOLD = 0.5  # Mudança de emoção para considerar anomalia (ajustado para reduzir falsos positivos)
HISTORY_SIZE = 5  # Tamanho da janela para análise de padrões
MIN_ACTIVITY_DURATION = 3  # Frames mínimos para considerar uma atividade
ACTIVITY_DISPLAY_DURATION = 30  # Frames para manter atividade visível no vídeo

# Constantes para validação de faces (filtrar bonecos/ilustrações)
MIN_FACE_WIDTH = 40  # Largura mínima da face em pixels (aumentado para melhor precisão)
MIN_FACE_HEIGHT = 40  # Altura mínima da face em pixels (aumentado para melhor precisão)
MIN_EMOTION_CONFIDENCE = 30.0  # Confiança mínima da emoção dominante (%) (aumentado)
MIN_ASPECT_RATIO = 0.6  # Proporção mínima largura/altura (ajustado para faces humanas)
MAX_ASPECT_RATIO = 1.8  # Proporção máxima largura/altura (ajustado para faces humanas)
MAX_EMOTION_ENTROPY = 0.80  # Entropia máxima das emoções (detecta distribuições muito uniformes = suspeito)
MIN_FACE_AREA = 1600  # Área mínima da face em pixels² (40x40)
MAX_FACE_AREA_RATIO = 0.5  # Máximo de área da face em relação ao frame (evita faces muito grandes = suspeito)


class VideoStatistics:
    """Classe para rastrear estatísticas durante o processamento do vídeo."""

    def __init__(self):
        self.frames_processed = 0
        self.total_faces_detected = 0
        self.faces_filtered = 0  # Contador de faces filtradas (bonecos/ilustrações)
        self.emotion_counts = defaultdict(int)
        self.activities_detected = []
        self.anomalies_detected = []
        self.face_positions_history = deque(maxlen=HISTORY_SIZE)
        self.emotion_history = deque(maxlen=HISTORY_SIZE)
        self.current_activity = None
        self.activity_start_frame = None
        self.active_activities = []  # Atividades ativas no frame atual para visualização

    def add_frame_data(
        self,
        frame_number: int,
        faces_data: List[Dict[str, Any]],
        previous_positions: Optional[List[Tuple[int, int, int, int]]] = None
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Adiciona dados de um frame e detecta atividades/anomalias.

        Args:
            frame_number: Número do frame atual
            faces_data: Lista de dados das faces detectadas
            previous_positions: Posições das faces no frame anterior

        Returns:
            Tupla com (lista de atividades detectadas, lista de anomalias detectadas)
        """
        self.frames_processed += 1
        activities = []
        anomalies = []

        current_positions = []
        current_emotions = []

        for face_data in faces_data:
            self.total_faces_detected += 1

            # Extrai emoção
            dominant_emotion = face_data.get('dominant_emotion')
            if dominant_emotion:
                self.emotion_counts[dominant_emotion] += 1
                current_emotions.append(dominant_emotion)

            # Extrai posição
            face_region = face_data.get('region', {})
            x = face_region.get('x', 0)
            y = face_region.get('y', 0)
            w = face_region.get('w', 0)
            h = face_region.get('h', 0)
            center_x = x + w // 2
            center_y = y + h // 2
            current_positions.append((center_x, center_y, w, h))

            # Detecta movimento/anomalia se houver posição anterior
            if previous_positions and len(previous_positions) > 0:
                # Calcula movimento em relação à face mais próxima
                min_distance = float('inf')
                closest_prev = None

                for prev_pos in previous_positions:
                    prev_center_x, prev_center_y = prev_pos[0], prev_pos[1]
                    distance = np.sqrt(
                        (center_x - prev_center_x) ** 2 +
                        (center_y - prev_center_y) ** 2
                    )
                    if distance < min_distance:
                        min_distance = distance
                        closest_prev = prev_pos

                if closest_prev:
                    movement = min_distance

                    # Detecta anomalia por movimento brusco
                    if movement > ANOMALY_MOVEMENT_THRESHOLD:
                        anomaly = {
                            'frame': frame_number,
                            'type': 'movimento_brusco',
                            'movement_pixels': float(movement),
                            'position': (center_x, center_y),
                            'description': f'Movimento brusco detectado: {movement:.1f} pixels'
                        }
                        anomalies.append(anomaly)
                        self.anomalies_detected.append(anomaly)

                    # Detecta atividade por movimento moderado
                    elif movement > MOVEMENT_THRESHOLD:
                        activity_type = self._categorize_activity(movement, dominant_emotion)
                        if self.current_activity != activity_type:
                            # Finaliza atividade anterior
                            if self.current_activity and self.activity_start_frame:
                                duration = frame_number - self.activity_start_frame
                                if duration >= MIN_ACTIVITY_DURATION:
                                    activity = {
                                        'type': self.current_activity,
                                        'start_frame': self.activity_start_frame,
                                        'end_frame': frame_number - 1,
                                        'duration_frames': duration
                                    }
                                    self.activities_detected.append(activity)
                                    activities.append(activity)

                            # Inicia nova atividade
                            self.current_activity = activity_type
                            self.activity_start_frame = frame_number

                        # Adiciona atividade ativa para visualização
                        if self.current_activity:
                            self.active_activities.append({
                                'type': self.current_activity,
                                'frame': frame_number,
                                'position': (center_x, center_y)
                            })

        # Detecta anomalia por mudança emocional brusca (apenas mudanças significativas)
        if len(self.emotion_history) >= 2 and current_emotions:
            # Compara com emoções de frames anteriores para detectar mudanças bruscas
            prev_emotions_set = set()
            for prev_emotions in list(self.emotion_history)[-2:]:  # Últimos 2 frames
                prev_emotions_set.update(prev_emotions)

            current_emotions_set = set(current_emotions)

            # Detecta apenas se houver mudança significativa (emoções completamente diferentes)
            if prev_emotions_set and current_emotions_set:
                # Calcula similaridade entre conjuntos de emoções
                intersection = prev_emotions_set & current_emotions_set
                union = prev_emotions_set | current_emotions_set

                # Se não houver sobreposição significativa, é uma mudança brusca
                if len(union) > 0:
                    similarity = len(intersection) / len(union)
                    if similarity < EMOTION_CHANGE_THRESHOLD:
                        anomaly = {
                            'frame': frame_number,
                            'type': 'mudanca_emocional_brusca',
                            'previous_emotions': list(prev_emotions_set),
                            'current_emotion': list(current_emotions_set),
                            'description': f'Mudança emocional brusca: {list(prev_emotions_set)} -> {list(current_emotions_set)}'
                        }
                        anomalies.append(anomaly)
                        self.anomalies_detected.append(anomaly)

        # Atualiza histórico
        self.face_positions_history.append(current_positions)
        self.emotion_history.append(set(current_emotions))

        return activities, anomalies

    def _categorize_activity(self, movement: float, emotion: Optional[str]) -> str:
        """
        Categoriza o tipo de atividade baseado no movimento e emoção.

        Categorias de atividades:
        - movimento_rapido: Movimentos rápidos e bruscos
        - movimento_moderado: Movimentos normais e moderados
        - gesto_expressivo: Gestos acompanhados de emoções positivas
        - gesto_intenso: Gestos acompanhados de emoções intensas
        - parado: Sem movimento significativo
        """
        if movement > 50:
            return 'movimento_rapido'
        elif movement > MOVEMENT_THRESHOLD:
            if emotion in ['happy', 'surprise']:
                return 'gesto_expressivo'
            elif emotion in ['angry', 'fear']:
                return 'gesto_intenso'
            else:
                return 'movimento_moderado'
        return 'parado'

    def finalize_activities(self, final_frame: int):
        """Finaliza atividades em andamento."""
        if self.current_activity and self.activity_start_frame:
            duration = final_frame - self.activity_start_frame
            if duration >= MIN_ACTIVITY_DURATION:
                activity = {
                    'type': self.current_activity,
                    'start_frame': self.activity_start_frame,
                    'end_frame': final_frame,
                    'duration_frames': duration
                }
                self.activities_detected.append(activity)

    def get_summary(self) -> Dict[str, Any]:
        """Gera um resumo das estatísticas coletadas."""
        # Conta atividades por tipo
        activity_counts = defaultdict(int)
        for activity in self.activities_detected:
            activity_counts[activity['type']] += 1

        # Emoções mais frequentes
        sorted_emotions = sorted(
            self.emotion_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return {
            'total_frames_analisados': self.frames_processed,
            'total_faces_detectadas': self.total_faces_detected,
            'faces_filtradas': self.faces_filtered,
            'numero_anomalias_detectadas': len(self.anomalies_detected),
            'anomalias': self.anomalies_detected,
            'atividades_detectadas': len(self.activities_detected),
            'atividades_por_tipo': dict(activity_counts),
            'detalhes_atividades': self.activities_detected,
            'emocoes_detectadas': dict(self.emotion_counts),
            'emocao_mais_frequente': sorted_emotions[0][0] if sorted_emotions else None,
            'distribuicao_emocoes': dict(sorted_emotions)
        }


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


def validate_face_detection(face_data: Dict[str, Any], frame_shape: Optional[Tuple[int, int]] = None) -> bool:
    """
    Valida se uma detecção de face é válida (não é boneco/ilustração).

    Implementa múltiplas validações para garantir que apenas faces humanas reais
    sejam aceitas, filtrando bonecos, ilustrações e detecções falsas.

    Args:
        face_data: Dicionário com dados da face detectada
        frame_shape: Tupla (altura, largura) do frame para validação de proporções

    Returns:
        True se a face é válida, False caso contrário
    """
    try:
        # Valida região da face
        face_region = face_data.get('region', {})
        w = face_region.get('w', 0)
        h = face_region.get('h', 0)

        # Valida dimensões mínimas
        if w < MIN_FACE_WIDTH or h < MIN_FACE_HEIGHT:
            return False

        # Valida área mínima
        face_area = w * h
        if face_area < MIN_FACE_AREA:
            return False

        # Valida proporção da face em relação ao frame (evita faces muito grandes = suspeito)
        if frame_shape:
            frame_height, frame_width = frame_shape
            frame_area = frame_width * frame_height
            if frame_area > 0:
                face_ratio = face_area / frame_area
                if face_ratio > MAX_FACE_AREA_RATIO:
                    return False

        # Valida proporção (aspect ratio) - faces humanas têm proporções específicas
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
            return False

        # Valida emoção e confiança
        dominant_emotion = face_data.get('dominant_emotion')
        emotion_dict = face_data.get('emotion', {})

        if not dominant_emotion or not emotion_dict:
            return False

        # Valida confiança mínima da emoção dominante
        emotion_confidence = emotion_dict.get(dominant_emotion, 0.0)
        if emotion_confidence < MIN_EMOTION_CONFIDENCE:
            return False

        # Valida distribuição de emoções (detecta distribuições muito uniformes = suspeito)
        # Faces humanas reais geralmente têm uma emoção dominante clara
        emotion_values = [v for v in emotion_dict.values() if v > 0]
        if len(emotion_values) > 1:
            total = sum(emotion_values)
            if total > 0:
                probabilities = [v / total for v in emotion_values]
                entropy = -sum(p * np.log2(p + 1e-10) for p in probabilities)
                max_entropy = np.log2(len(probabilities))
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

                # Se a entropia for muito alta, significa que as emoções estão muito uniformes
                # Isso pode indicar uma detecção inválida (boneco/ilustração)
                if normalized_entropy > MAX_EMOTION_ENTROPY:
                    return False

                # Valida se há uma emoção claramente dominante (diferença mínima)
                sorted_emotions = sorted(emotion_values, reverse=True)
                if len(sorted_emotions) >= 2:
                    dominant_ratio = sorted_emotions[0] / sorted_emotions[1] if sorted_emotions[1] > 0 else 0
                    # Se a emoção dominante não for pelo menos 1.3x maior que a segunda, é suspeito
                    if dominant_ratio < 1.3:
                        return False

        # Valida se a emoção dominante está em um conjunto válido de emoções humanas
        valid_emotions = {'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'}
        if dominant_emotion not in valid_emotions:
            return False

        return True
    except Exception as e:
        logger.warning(f"Erro ao validar face: {e}")
        return False


def analyze_frame(frame: cv2.typing.MatLike, stats: Optional['VideoStatistics'] = None) -> List[Dict[str, Any]]:
    """
    Analisa um frame detectando faces e emoções, filtrando detecções inválidas.

    Usa um backend robusto de detecção (RetinaFace por padrão) que é mais preciso
    em detectar apenas faces humanas reais, reduzindo falsos positivos.

    Args:
        frame: Frame do vídeo a ser analisado
        stats: Objeto VideoStatistics opcional para contar faces filtradas

    Returns:
        Lista de dicionários com dados das faces detectadas (apenas válidas)
    """
    try:
        # Tenta usar o backend configurado, com fallback para opencv se falhar
        backend = DETECTOR_BACKEND
        try:
            results = DeepFace.analyze(
                frame,
                actions=EMOTION_ACTIONS,
                enforce_detection=False,
                detector_backend=backend,
                silent=True
            )
        except Exception as backend_error:
            # Fallback para opencv se o backend configurado falhar
            logger.warning(f"Backend {backend} falhou, usando opencv como fallback: {backend_error}")
            backend = 'opencv'
            results = DeepFace.analyze(
                frame,
                actions=EMOTION_ACTIONS,
                enforce_detection=False,
                detector_backend=backend,
                silent=True
            )

        # Garante que results seja sempre uma lista
        if not isinstance(results, list):
            results = [results]

        # Obtém dimensões do frame para validação
        frame_shape = frame.shape[:2] if frame is not None else None

        # Filtra detecções inválidas (bonecos, ilustrações, etc.)
        valid_results = []
        for face_data in results:
            if validate_face_detection(face_data, frame_shape):
                valid_results.append(face_data)
            else:
                if stats:
                    stats.faces_filtered += 1
                logger.debug("Face inválida filtrada (possível boneco/ilustração)")

        return valid_results
    except Exception as e:
        logger.warning(f"Erro ao analisar frame: {e}")
        return []


def draw_anomalies(frame: cv2.typing.MatLike, anomalies: List[Dict[str, Any]]) -> None:
    """
    Desenha indicadores visuais de anomalias no frame.

    Args:
        frame: Frame onde desenhar as anomalias
        anomalies: Lista de anomalias detectadas
    """
    for anomaly in anomalies:
        if 'position' in anomaly:
            pos = anomaly['position']
            cv2.circle(frame, pos, 15, ANOMALY_COLOR, 3)
            cv2.putText(
                frame,
                "ANOMALIA",
                (pos[0] - 40, pos[1] - 20),
                DEFAULT_FONT,
                0.5,
                ANOMALY_COLOR,
                2
            )


def draw_activities(frame: cv2.typing.MatLike, activities: List[Dict[str, Any]], frame_number: int) -> None:
    """
    Desenha indicadores visuais de atividades no frame.

    Args:
        frame: Frame onde desenhar as atividades
        activities: Lista de atividades ativas
        frame_number: Número do frame atual
    """
    # Remove atividades antigas (fora da janela de visualização)
    active_activities = [
        act for act in activities
        if frame_number - act['frame'] <= ACTIVITY_DISPLAY_DURATION
    ]

    # Desenha atividades ativas
    for activity in active_activities:
        if 'position' in activity:
            pos = activity['position']
            activity_type = activity['type']

            # Mapeia tipos de atividade para nomes mais descritivos
            activity_names = {
                'movimento_rapido': 'Movimento Rapido',
                'movimento_moderado': 'Movimento Moderado',
                'gesto_expressivo': 'Gesto Expressivo',
                'gesto_intenso': 'Gesto Intenso',
                'parado': 'Parado'
            }

            display_name = activity_names.get(activity_type, activity_type.replace('_', ' ').title())

            # Desenha indicador de atividade
            cv2.rectangle(
                frame,
                (pos[0] - 60, pos[1] + 30),
                (pos[0] + 60, pos[1] + 50),
                ACTIVITY_COLOR,
                -1
            )
            cv2.putText(
                frame,
                display_name,
                (pos[0] - 55, pos[1] + 45),
                DEFAULT_FONT,
                0.4,
                (255, 255, 255),
                1
            )


def process_single_frame(
    frame: cv2.typing.MatLike,
    display: bool,
    out: Optional[cv2.VideoWriter],
    anomalies: Optional[List[Dict[str, Any]]] = None,
    activities: Optional[List[Dict[str, Any]]] = None,
    frame_number: int = 0
) -> Tuple[int, List[Dict[str, Any]]]:
    """
    Processa um único frame do vídeo.

    Args:
        frame: Frame a ser processado
        display: Se True, exibe o frame
        out: Writer de vídeo para salvar o frame
        anomalies: Lista de anomalias detectadas neste frame
        activities: Lista de atividades ativas para visualização
        frame_number: Número do frame atual

    Returns:
        Tupla com (número de faces detectadas, lista de dados das faces)
        ou (-1, []) se interrompido pelo usuário
    """
    faces_detected = 0
    results = analyze_frame(frame, stats=None)

    for face_data in results:
        process_face_detection(frame, face_data)
        faces_detected += 1

    # Desenha indicadores de atividades
    if activities:
        draw_activities(frame, activities, frame_number)

    # Desenha indicadores de anomalias
    if anomalies:
        draw_anomalies(frame, anomalies)

    if display:
        cv2.imshow('Video - Pressione Q para sair', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info("Processamento interrompido pelo usuário")
            return -1, []  # Indica interrupção

    if out:
        out.write(frame)

    return faces_detected, results


def generate_report(summary: Dict[str, Any], output_path: Optional[str] = None) -> str:
    """
    Gera um relatório em texto a partir do resumo das estatísticas.

    Args:
        summary: Dicionário com resumo das estatísticas
        output_path: Caminho opcional para salvar o relatório

    Returns:
        String com o relatório formatado
    """
    # Prepara dados para resumo executivo
    top_emotions = sorted(
        summary['distribuicao_emocoes'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:3]  # Top 3 emoções

    top_activities = sorted(
        summary['atividades_por_tipo'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:3]  # Top 3 atividades

    report_lines = [
        "=" * 80,
        "RELATÓRIO DE ANÁLISE DE VÍDEO",
        "=" * 80,
        "",
        f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "RESUMO EXECUTIVO",
        "-" * 80,
        "Este relatório apresenta uma análise completa do vídeo, incluindo:",
        "- Reconhecimento facial e marcação de rostos",
        "- Análise de expressões emocionais",
        "- Detecção e categorização de atividades",
        "- Identificação de anomalias (movimentos bruscos e comportamentos atípicos)",
        "",
        "PRINCIPAIS RESULTADOS:",
        f"  • Total de frames analisados: {summary['total_frames_analisados']}",
        f"  • Número de anomalias detectadas: {summary['numero_anomalias_detectadas']}",
        "",
        "PRINCIPAIS EMOÇÕES DETECTADAS:",
    ]

    for i, (emotion, count) in enumerate(top_emotions, 1):
        percentage = (count / summary['total_faces_detectadas'] * 100) if summary['total_faces_detectadas'] > 0 else 0
        report_lines.append(f"  {i}. {emotion.capitalize()}: {count} ocorrências ({percentage:.1f}%)")

    if top_activities:
        report_lines.append("")
        report_lines.append("PRINCIPAIS ATIVIDADES DETECTADAS:")
        for i, (activity, count) in enumerate(top_activities, 1):
            report_lines.append(f"  {i}. {activity.replace('_', ' ').title()}: {count} ocorrências")

    report_lines.extend([
        "",
        "=" * 80,
        "DETALHES COMPLETOS",
        "=" * 80,
        "",
        "RESUMO GERAL",
        "-" * 80,
        f"Total de frames analisados: {summary['total_frames_analisados']}",
        f"Total de faces detectadas: {summary['total_faces_detectadas']}",
        f"Faces filtradas (bonecos/ilustrações): {summary.get('faces_filtradas', 0)}",
        f"Número de anomalias detectadas: {summary['numero_anomalias_detectadas']}",
        "",
        "ANÁLISE DE EMOÇÕES",
        "-" * 80,
    ])

    # Adiciona distribuição de emoções
    for emotion, count in summary['distribuicao_emocoes'].items():
        percentage = (count / summary['total_faces_detectadas'] * 100) if summary['total_faces_detectadas'] > 0 else 0
        report_lines.append(f"  {emotion.capitalize()}: {count} ocorrências ({percentage:.1f}%)")

    if summary['emocao_mais_frequente']:
        report_lines.append(f"\n  Emoção mais frequente: {summary['emocao_mais_frequente'].capitalize()}")

    report_lines.extend([
        "",
        "ANÁLISE DE ATIVIDADES",
        "-" * 80,
        f"Total de atividades detectadas: {summary['atividades_detectadas']}",
    ])

    # Adiciona atividades por tipo
    if summary['atividades_por_tipo']:
        report_lines.append("\n  Atividades por tipo:")
        for activity_type, count in summary['atividades_por_tipo'].items():
            report_lines.append(f"    - {activity_type.replace('_', ' ').title()}: {count}")

    # Adiciona detalhes das atividades
    if summary['detalhes_atividades']:
        report_lines.append("\n  Detalhes das atividades:")
        for i, activity in enumerate(summary['detalhes_atividades'][:10], 1):  # Limita a 10
            report_lines.append(
                f"    {i}. {activity['type'].replace('_', ' ').title()} "
                f"(frames {activity['start_frame']}-{activity['end_frame']}, "
                f"duração: {activity['duration_frames']} frames)"
            )

    report_lines.extend([
        "",
        "DETECÇÃO DE ANOMALIAS",
        "-" * 80,
    ])

    if summary['anomalias']:
        # Agrupa anomalias por tipo
        anomaly_types = defaultdict(int)
        for anomaly in summary['anomalias']:
            anomaly_types[anomaly['type']] += 1

        report_lines.append("  Anomalias por tipo:")
        for anomaly_type, count in anomaly_types.items():
            report_lines.append(f"    - {anomaly_type.replace('_', ' ').title()}: {count}")

        report_lines.append("\n  Detalhes das anomalias (primeiras 10):")
        for i, anomaly in enumerate(summary['anomalias'][:10], 1):
            report_lines.append(f"    {i}. Frame {anomaly['frame']}: {anomaly.get('description', 'Anomalia detectada')}")
    else:
        report_lines.append("  Nenhuma anomalia detectada.")

    report_lines.extend([
        "",
        "=" * 80,
        "Fim do Relatório",
        "=" * 80,
    ])

    report_text = "\n".join(report_lines)

    # Salva o relatório se um caminho foi fornecido
    if output_path:
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"Relatório salvo em: {output_path}")
        except Exception as e:
            logger.error(f"Erro ao salvar relatório: {e}")

    return report_text


def detect_expressions_in_video(
    video_path: str,
    output_path: Optional[str] = None,
    display: bool = False,
    report_path: Optional[str] = None
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Detecta expressões faciais, emoções, atividades e anomalias em um vídeo.

    Processa o vídeo frame a frame, detectando faces, identificando emoções,
    categorizando atividades e detectando anomalias, gerando um vídeo de saída
    com anotações visuais opcionais e um relatório completo.

    Args:
        video_path: Caminho para o arquivo de vídeo de entrada
        output_path: Caminho opcional para salvar o vídeo processado.
                    Se None, o vídeo não será salvo
        display: Se True, exibe o vídeo em tempo real durante o processamento
        report_path: Caminho opcional para salvar o relatório de análise

    Returns:
        Tupla com (True/False indicando sucesso, dicionário com resumo das estatísticas)

    Raises:
        FileNotFoundError: Se o arquivo de vídeo não existir
        ValueError: Se os parâmetros forem inválidos
    """
    # Validação de entrada
    validate_video_path(video_path)
    validate_output_path(output_path)
    if report_path:
        validate_output_path(report_path)

    logger.info(f"Iniciando processamento do vídeo: {video_path}")

    # Abre o vídeo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        error_msg = f"Não foi possível abrir o vídeo: {video_path}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Inicializa estatísticas
    stats = VideoStatistics()
    previous_positions = None

    out = None
    try:
        # Obtém propriedades do vídeo
        props = get_video_properties(cap)
        if not props:
            return False, None

        # Configura o writer de vídeo se necessário
        if output_path:
            out = setup_video_writer(
                output_path,
                props['fps'],
                props['width'],
                props['height']
            )
            if not out:
                return False, None

        # Processa cada frame
        with tqdm(total=props['frame_count'], desc="Processando frames do vídeo") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Analisa o frame para obter dados das faces (filtra bonecos/ilustrações)
                faces_data = analyze_frame(frame, stats=stats)

                # Extrai posições das faces para análise de movimento
                current_positions = []
                for face_data_item in faces_data:
                    face_region = face_data_item.get('region', {})
                    x = face_region.get('x', 0)
                    y = face_region.get('y', 0)
                    w = face_region.get('w', 0)
                    h = face_region.get('h', 0)
                    center_x = x + w // 2
                    center_y = y + h // 2
                    current_positions.append((center_x, center_y, w, h))

                # Adiciona dados ao sistema de estatísticas e detecta atividades/anomalias
                frame_number = stats.frames_processed + 1
                _, anomalies = stats.add_frame_data(
                    frame_number,
                    faces_data,
                    previous_positions
                )

                # Limpa atividades antigas da lista de visualização (otimização de memória)
                stats.active_activities = [
                    act for act in stats.active_activities
                    if frame_number - act['frame'] <= ACTIVITY_DISPLAY_DURATION
                ]

                # Processa o frame com todas as anotações (faces, emoções, atividades e anomalias)
                faces_in_frame, _ = process_single_frame(
                    frame,
                    display,
                    out,
                    anomalies=anomalies,
                    activities=stats.active_activities,
                    frame_number=frame_number
                )

                if faces_in_frame == -1:  # Interrupção pelo usuário
                    break

                previous_positions = current_positions
                pbar.update(1)

        # Finaliza atividades em andamento
        stats.finalize_activities(stats.frames_processed)

        # Gera resumo
        summary = stats.get_summary()

        logger.info(
            f"Processamento concluído: {summary['total_frames_analisados']} frames processados, "
            f"{summary['total_faces_detectadas']} faces detectadas, "
            f"{summary['numero_anomalias_detectadas']} anomalias detectadas"
        )

        if output_path:
            logger.info(f"Vídeo de saída salvo em: {output_path}")

        # Gera e salva relatório
        report_text = generate_report(summary, report_path)
        if report_path:
            logger.info(f"Relatório salvo em: {report_path}")
        else:
            # Exibe relatório no console se não foi especificado caminho
            logger.info("\n" + report_text)

        return True, summary

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
        report_file = script_dir / "relatorio_analise.txt"

        success, summary = detect_expressions_in_video(
            str(video_file),
            str(output_file),
            display=False,
            report_path=str(report_file)
        )

        if success:
            logger.info("Processamento concluído com sucesso!")
            if summary:
                logger.info(
                    f"Resumo: {summary['total_frames_analisados']} frames, "
                    f"{summary['numero_anomalias_detectadas']} anomalias, "
                    f"{summary['atividades_detectadas']} atividades"
                )
        else:
            logger.error("Processamento falhou")

    except Exception as e:
        logger.error(f"Erro durante a execução: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()