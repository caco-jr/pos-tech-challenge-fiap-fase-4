"""
Módulo para detecção de expressões faciais e emoções em vídeos.

Este módulo utiliza DeepFace e OpenCV para processar vídeos frame a frame,
detectando faces e identificando as emoções dominantes, gerando um vídeo
de saída com anotações visuais.
"""

import cv2
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from collections import defaultdict, deque
from datetime import datetime
from tqdm import tqdm
import numpy as np
import mediapipe as mp

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

# Constantes para detecção de pose e atividades específicas
MIN_POSE_CONFIDENCE = 0.5
HAND_FACE_DISTANCE_THRESHOLD = 0.15
WAVE_MOVEMENT_THRESHOLD = 0.05
WAVE_FRAMES_THRESHOLD = 5
VALID_EMOTIONS = {'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'}

# Constantes para validação de faces (filtrar bonecos/ilustrações)
MIN_FACE_WIDTH = 40  # Largura mínima da face em pixels (aumentado para melhor precisão)
MIN_FACE_HEIGHT = 40  # Altura mínima da face em pixels (aumentado para melhor precisão)
MIN_EMOTION_CONFIDENCE = 30.0  # Confiança mínima da emoção dominante (%) (aumentado)
MIN_ASPECT_RATIO = 0.6  # Proporção mínima largura/altura (ajustado para faces humanas)
MAX_ASPECT_RATIO = 1.8  # Proporção máxima largura/altura (ajustado para faces humanas)
MAX_EMOTION_ENTROPY = 0.80  # Entropia máxima das emoções (detecta distribuições muito uniformes = suspeito)
MIN_FACE_AREA = 1600  # Área mínima da face em pixels² (40x40)
MAX_FACE_AREA_RATIO = 0.5  # Máximo de área da face em relação ao frame (evita faces muito grandes = suspeito)


def _get_mp_pose():
    """Obtém módulo MediaPipe Pose de forma compatível."""
    try:
        return mp.solutions.pose if hasattr(mp, 'solutions') else None
    except AttributeError:
        return None


def _get_landmark(landmarks, landmark_idx):
    """Extrai landmark de forma compatível com diferentes versões do MediaPipe."""
    try:
        idx = landmark_idx.value if hasattr(landmark_idx, 'value') else landmark_idx
        return landmarks[idx]
    except (AttributeError, TypeError, IndexError):
        return None


def analyze_pose_for_activity(landmarks: Any) -> Dict[str, Any]:
    """Analisa landmarks de pose para extrair informações sobre a postura corporal."""
    mp_pose = _get_mp_pose()
    if not mp_pose:
        return _default_pose_info()

    pose_info = _default_pose_info()

    try:
        # Extrai landmarks principais
        left_shoulder = _get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
        right_shoulder = _get_landmark(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER)
        left_elbow = _get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW)
        right_elbow = _get_landmark(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW)
        left_wrist = _get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_WRIST)
        right_wrist = _get_landmark(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST)
        left_hip = _get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
        right_hip = _get_landmark(landmarks, mp_pose.PoseLandmark.RIGHT_HIP)
        left_knee = _get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_KNEE)
        right_knee = _get_landmark(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE)
        nose = _get_landmark(landmarks, mp_pose.PoseLandmark.NOSE)
        left_ankle = _get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)
        right_ankle = _get_landmark(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE)

        if not all([left_shoulder, right_shoulder, left_elbow, right_elbow,
                   left_wrist, right_wrist, left_hip, right_hip, left_knee,
                   right_knee, nose, left_ankle, right_ankle]):
            return pose_info

        # Verifica braços levantados
        left_arm_up = left_wrist.y < left_elbow.y < left_shoulder.y
        right_arm_up = right_wrist.y < right_elbow.y < right_shoulder.y
        pose_info.update({
            'left_arm_up': left_arm_up,
            'right_arm_up': right_arm_up,
            'hands_raised': left_arm_up or right_arm_up
        })

        # Determina posição dos braços
        if left_arm_up or right_arm_up:
            pose_info['arms_position'] = 'up'
        elif left_wrist.y > left_shoulder.y and right_wrist.y > right_shoulder.y:
            pose_info['arms_position'] = 'down'

        # Verifica postura
        hip_y = (left_hip.y + right_hip.y) / 2
        knee_y = (left_knee.y + right_knee.y) / 2
        ankle_y = (left_ankle.y + right_ankle.y) / 2
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2

        if abs(hip_y - knee_y) < 0.15 and abs(knee_y - ankle_y) > 0.1:
            pose_info['is_sitting'] = True
        if abs(shoulder_y - hip_y) < 0.1:
            pose_info['is_lying'] = True
        if nose.y > left_shoulder.y and nose.y > right_shoulder.y:
            pose_info['head_tilted_down'] = True

        # Verifica mão próxima ao rosto
        left_dist = np.sqrt((left_wrist.x - nose.x) ** 2 + (left_wrist.y - nose.y) ** 2)
        right_dist = np.sqrt((right_wrist.x - nose.x) ** 2 + (right_wrist.y - nose.y) ** 2)
        if left_dist < HAND_FACE_DISTANCE_THRESHOLD or right_dist < HAND_FACE_DISTANCE_THRESHOLD:
            pose_info['hand_near_face'] = True

    except (IndexError, AttributeError, KeyError) as e:
        logger.debug(f"Erro ao analisar pose: {e}")

    return pose_info


def _default_pose_info() -> Dict[str, Any]:
    """Retorna informações de pose padrão."""
    return {
        'hands_raised': False,
        'is_sitting': False,
        'head_tilted_down': False,
        'is_lying': False,
        'hand_near_face': False,
        'arms_position': 'neutral',
        'left_arm_up': False,
        'right_arm_up': False
    }


def _extract_face_positions(faces_data: List[Dict[str, Any]]) -> List[Tuple[int, int, int, int]]:
    """Extrai posições das faces detectadas."""
    positions = []
    for face_data in faces_data:
        region = face_data.get('region', {})
        x, y = region.get('x', 0), region.get('y', 0)
        w, h = region.get('w', 0), region.get('h', 0)
        positions.append((x + w // 2, y + h // 2, w, h))
    return positions


def _create_activity(activity_type: str, start_frame: int, end_frame: int) -> Dict[str, Any]:
    """Cria dicionário de atividade."""
    return {
        'type': activity_type,
        'start_frame': start_frame,
        'end_frame': end_frame,
        'duration_frames': end_frame - start_frame + 1
    }


class VideoStatistics:
    """Classe para rastrear estatísticas durante o processamento do vídeo."""

    def __init__(self):
        self.frames_processed = 0
        self.total_faces_detected = 0
        self.faces_filtered = 0
        self.emotion_counts = defaultdict(int)
        self.activities_detected = []
        self.anomalies_detected = []
        self.face_positions_history = deque(maxlen=HISTORY_SIZE)
        self.emotion_history = deque(maxlen=HISTORY_SIZE)
        self.current_activity = None
        self.activity_start_frame = None
        self.active_activities = []
        self.prev_left_wrist_pos = None
        self.prev_right_wrist_pos = None
        self.wave_frames = 0

    def add_frame_data(
        self,
        frame_number: int,
        faces_data: List[Dict[str, Any]],
        previous_positions: Optional[List[Tuple[int, int, int, int]]] = None,
        pose_info: Optional[Dict[str, Any]] = None,
        pose_landmarks: Optional[Any] = None
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Adiciona dados de um frame e detecta atividades/anomalias."""
        self.frames_processed += 1
        activities = []
        anomalies = []

        # Extrai dados das faces
        current_positions = _extract_face_positions(faces_data)
        current_emotions = []
        dominant_emotion = None

        for face_data in faces_data:
            self.total_faces_detected += 1
            emotion = face_data.get('dominant_emotion')
            if emotion:
                self.emotion_counts[emotion] += 1
                current_emotions.append(emotion)
                dominant_emotion = emotion

        # Detecta atividades e anomalias
        if pose_info:
            self._process_pose_based_activity(frame_number, pose_info, dominant_emotion,
                                             pose_landmarks, current_positions, activities)
        elif previous_positions and current_positions:
            self._process_movement_based_activity(frame_number, previous_positions,
                                                current_positions, dominant_emotion,
                                                activities, anomalies)

        # Detecta anomalias emocionais
        self._detect_emotion_anomalies(frame_number, current_emotions, anomalies)

        # Atualiza histórico
        self.face_positions_history.append(current_positions)
        self.emotion_history.append(set(current_emotions))

        return activities, anomalies

    def _process_pose_based_activity(self, frame_number: int, pose_info: Dict[str, Any],
                                    emotion: Optional[str], landmarks: Optional[Any],
                                    positions: List[Tuple], activities: List[Dict]):
        """Processa atividades baseadas em pose."""
        activity_type = self._categorize_activity_from_pose(pose_info, emotion, landmarks)

        if activity_type and activity_type != self.current_activity:
            self._finalize_current_activity(frame_number, activities)
            self.current_activity = activity_type
            self.activity_start_frame = frame_number

        if self.current_activity and positions:
            self.active_activities.append({
                'type': self.current_activity,
                'frame': frame_number,
                'position': positions[0][:2]
            })

    def _process_movement_based_activity(self, frame_number: int,
                                        previous_positions: List[Tuple],
                                        current_positions: List[Tuple],
                                        emotion: Optional[str],
                                        activities: List[Dict],
                                        anomalies: List[Dict]):
        """Processa atividades baseadas em movimento."""
        movement = self._calculate_movement(previous_positions, current_positions)

        if movement > ANOMALY_MOVEMENT_THRESHOLD:
            anomalies.append(self._create_movement_anomaly(frame_number, movement, current_positions))
        elif movement > MOVEMENT_THRESHOLD:
            activity_type = self._categorize_activity(movement, emotion)
            if activity_type != self.current_activity:
                self._finalize_current_activity(frame_number, activities)
                self.current_activity = activity_type
                self.activity_start_frame = frame_number

            if self.current_activity and current_positions:
                self.active_activities.append({
                    'type': self.current_activity,
                    'frame': frame_number,
                    'position': current_positions[0][:2]
                })

    def _calculate_movement(self, prev_positions: List[Tuple],
                          curr_positions: List[Tuple]) -> float:
        """Calcula movimento mínimo entre posições."""
        min_distance = float('inf')
        for prev_pos in prev_positions:
            for curr_pos in curr_positions:
                distance = np.sqrt(
                    (curr_pos[0] - prev_pos[0]) ** 2 + (curr_pos[1] - prev_pos[1]) ** 2
                )
                min_distance = min(min_distance, distance)
        return min_distance if min_distance != float('inf') else 0.0

    def _finalize_current_activity(self, frame_number: int, activities: List[Dict]):
        """Finaliza atividade atual se atender critérios mínimos."""
        if self.current_activity and self.activity_start_frame:
            duration = frame_number - self.activity_start_frame
            if duration >= MIN_ACTIVITY_DURATION:
                activity = _create_activity(
                    self.current_activity,
                    self.activity_start_frame,
                    frame_number - 1
                )
                self.activities_detected.append(activity)
                activities.append(activity)

    def _create_movement_anomaly(self, frame_number: int, movement: float,
                                positions: List[Tuple]) -> Dict[str, Any]:
        """Cria anomalia de movimento brusco."""
        anomaly = {
            'frame': frame_number,
            'type': 'movimento_brusco',
            'movement_pixels': float(movement),
            'position': positions[0][:2],
            'description': f'Movimento brusco detectado: {movement:.1f} pixels'
        }
        self.anomalies_detected.append(anomaly)
        return anomaly

    def _detect_emotion_anomalies(self, frame_number: int, current_emotions: List[str],
                                 anomalies: List[Dict]):
        """Detecta anomalias por mudança emocional brusca."""
        if len(self.emotion_history) < 2 or not current_emotions:
            return

        prev_emotions_set = set()
        for prev_emotions in list(self.emotion_history)[-2:]:
            prev_emotions_set.update(prev_emotions)

        current_emotions_set = set(current_emotions)
        if not prev_emotions_set or not current_emotions_set:
            return

        intersection = prev_emotions_set & current_emotions_set
        union = prev_emotions_set | current_emotions_set

        if union and len(intersection) / len(union) < EMOTION_CHANGE_THRESHOLD:
            anomaly = {
                'frame': frame_number,
                'type': 'mudanca_emocional_brusca',
                'previous_emotions': list(prev_emotions_set),
                'current_emotion': list(current_emotions_set),
                'description': f'Mudança emocional brusca: {list(prev_emotions_set)} -> {list(current_emotions_set)}'
            }
            anomalies.append(anomaly)
            self.anomalies_detected.append(anomaly)

    def _categorize_activity_from_pose(
        self,
        pose_info: Dict[str, Any],
        emotion: Optional[str],
        landmarks: Optional[Any] = None
    ) -> Optional[str]:
        """Categoriza atividade baseado em análise de pose corporal."""
        # Detecta aceno (tchau)
        if pose_info.get('hands_raised') and landmarks:
            if self._detect_wave(landmarks):
                return 'acenando_tchau'

        # Mãos levantadas
        if pose_info.get('hands_raised'):
            left_arm = pose_info.get('left_arm_up', False)
            right_arm = pose_info.get('right_arm_up', False)
            if left_arm and right_arm:
                return 'maos_levantadas_ambas'
            elif left_arm:
                return 'mao_levantada_esquerda'
            elif right_arm:
                return 'mao_levantada_direita'

        # Outras atividades
        if pose_info.get('is_lying'):
            return 'procedimento_estetico'
        if pose_info.get('hand_near_face'):
            return 'expressao_surpresa' if emotion == 'surprise' else 'refletindo'
        if pose_info.get('is_sitting') and pose_info.get('head_tilted_down'):
            return 'usando_celular_ou_lendo' if pose_info.get('arms_position') == 'down' else 'estudando'
        if pose_info.get('is_sitting') and pose_info.get('arms_position') == 'down':
            if emotion in ['neutral', 'happy', None]:
                return 'trabalhando'

        return None

    def _detect_wave(self, landmarks: Any) -> bool:
        """Detecta movimento de aceno (tchau)."""
        try:
            mp_pose = _get_mp_pose()
            if not mp_pose:
                return False

            left_wrist = _get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_WRIST)
            right_wrist = _get_landmark(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST)

            if not left_wrist or not right_wrist:
                return False

            # Verifica movimento lateral
            if self.prev_left_wrist_pos:
                movement = abs(left_wrist.x - self.prev_left_wrist_pos[0])
                self.wave_frames += 1 if movement > WAVE_MOVEMENT_THRESHOLD else -1
                self.wave_frames = max(0, self.wave_frames)

            if self.prev_right_wrist_pos:
                movement = abs(right_wrist.x - self.prev_right_wrist_pos[0])
                self.wave_frames += 1 if movement > WAVE_MOVEMENT_THRESHOLD else -1
                self.wave_frames = max(0, self.wave_frames)

            # Atualiza posições
            self.prev_left_wrist_pos = (left_wrist.x, left_wrist.y)
            self.prev_right_wrist_pos = (right_wrist.x, right_wrist.y)

            return self.wave_frames >= WAVE_FRAMES_THRESHOLD
        except (AttributeError, IndexError, TypeError):
            return False

    def _categorize_activity(self, movement: float, emotion: Optional[str]) -> str:
        """
        Categoriza o tipo de atividade baseado no movimento e emoção (fallback).

        Usado quando não há informações de pose disponíveis.

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
                activity = _create_activity(
                    self.current_activity,
                    self.activity_start_frame,
                    final_frame
                )
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
        if dominant_emotion not in VALID_EMOTIONS:
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
                # Atividades baseadas em pose (prioridade alta)
                'maos_levantadas_ambas': 'Mãos Levantadas',
                'mao_levantada_esquerda': 'Mão Esquerda Levantada',
                'mao_levantada_direita': 'Mão Direita Levantada',
                'acenando_tchau': 'Acenando (Tchau)',
                'procedimento_estetico': 'Procedimento Estético / Repouso',
                'refletindo': 'Refletindo',
                'expressao_surpresa': 'Expressão de Surpresa',
                'estudando': 'Estudando',
                'usando_celular_ou_lendo': 'Usando Celular / Lendo',
                'trabalhando': 'Trabalhando / Digitando',
                # Atividades baseadas em movimento (fallback)
                'movimento_rapido': 'Movimento Rápido',
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
    anomalies: Optional[List[Dict[str, Any]]] = None
) -> Tuple[int, List[Dict[str, Any]]]:
    """
    Processa um único frame do vídeo.

    Args:
        frame: Frame a ser processado
        display: Se True, exibe o frame
        out: Writer de vídeo para salvar o frame
        anomalies: Lista de anomalias detectadas neste frame

    Returns:
        Tupla com (número de faces detectadas, lista de dados das faces)
        ou (-1, []) se interrompido pelo usuário
    """
    faces_detected = 0
    results = analyze_frame(frame, stats=None)

    for face_data in results:
        process_face_detection(frame, face_data)
        faces_detected += 1

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


def _format_percentage(count: int, total: int) -> float:
    """Calcula porcentagem."""
    return (count / total * 100) if total > 0 else 0.0


def _format_activity_name(activity_type: str) -> str:
    """Formata nome da atividade para exibição."""
    return activity_type.replace('_', ' ').title()


def _init_mediapipe_pose():
    """Inicializa MediaPipe Pose de forma compatível."""
    try:
        if hasattr(mp, 'solutions'):
            mp_pose = mp.solutions.pose
            mp_drawing = mp.solutions.drawing_utils
        else:
            from mediapipe.python.solutions import pose as mp_pose
            from mediapipe.python.solutions import drawing_utils as mp_drawing

        pose = mp_pose.Pose(
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=MIN_POSE_CONFIDENCE,
            min_tracking_confidence=MIN_POSE_CONFIDENCE,
            static_image_mode=False
        )
        return pose, mp_pose, mp_drawing
    except Exception as e:
        logger.warning(f"MediaPipe Pose não disponível: {e}. Continuando sem detecção de pose.")
        return None, None, None


def generate_report(summary: Dict[str, Any], output_path: Optional[str] = None) -> str:
    """Gera um relatório em texto a partir do resumo das estatísticas."""
    total_faces = summary['total_faces_detectadas']

    # Top 3 emoções e atividades
    top_emotions = sorted(summary['distribuicao_emocoes'].items(),
                         key=lambda x: x[1], reverse=True)[:3]
    top_activities = sorted(summary['atividades_por_tipo'].items(),
                           key=lambda x: x[1], reverse=True)[:3]

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
        pct = _format_percentage(count, total_faces)
        report_lines.append(f"  {i}. {emotion.capitalize()}: {count} ocorrências ({pct:.1f}%)")

    if top_activities:
        report_lines.extend(["", "PRINCIPAIS ATIVIDADES DETECTADAS:"])
        for i, (activity, count) in enumerate(top_activities, 1):
            report_lines.append(f"  {i}. {_format_activity_name(activity)}: {count} ocorrências")

    report_lines.extend([
        "",
        "=" * 80,
        "DETALHES COMPLETOS",
        "=" * 80,
        "",
        "RESUMO GERAL",
        "-" * 80,
        f"Total de frames analisados: {summary['total_frames_analisados']}",
        f"Total de faces detectadas: {total_faces}",
        f"Faces filtradas (bonecos/ilustrações): {summary.get('faces_filtradas', 0)}",
        f"Número de anomalias detectadas: {summary['numero_anomalias_detectadas']}",
        "",
        "ANÁLISE DE EMOÇÕES",
        "-" * 80,
    ])

    for emotion, count in summary['distribuicao_emocoes'].items():
        pct = _format_percentage(count, total_faces)
        report_lines.append(f"  {emotion.capitalize()}: {count} ocorrências ({pct:.1f}%)")

    if summary['emocao_mais_frequente']:
        report_lines.append(f"\n  Emoção mais frequente: {summary['emocao_mais_frequente'].capitalize()}")

    report_lines.extend([
        "",
        "ANÁLISE DE ATIVIDADES",
        "-" * 80,
        f"Total de atividades detectadas: {summary['atividades_detectadas']}",
    ])

    if summary['atividades_por_tipo']:
        report_lines.append("\n  Atividades por tipo:")
        for activity_type, count in summary['atividades_por_tipo'].items():
            report_lines.append(f"    - {_format_activity_name(activity_type)}: {count}")

    if summary['detalhes_atividades']:
        report_lines.append("\n  Detalhes das atividades:")
        for i, activity in enumerate(summary['detalhes_atividades'][:10], 1):
            report_lines.append(
                f"    {i}. {_format_activity_name(activity['type'])} "
                f"(frames {activity['start_frame']}-{activity['end_frame']}, "
                f"duração: {activity['duration_frames']} frames)"
            )

    report_lines.extend(["", "DETECÇÃO DE ANOMALIAS", "-" * 80])

    if summary['anomalias']:
        anomaly_types = defaultdict(int)
        for anomaly in summary['anomalias']:
            anomaly_types[anomaly['type']] += 1

        report_lines.append("  Anomalias por tipo:")
        for anomaly_type, count in anomaly_types.items():
            report_lines.append(f"    - {_format_activity_name(anomaly_type)}: {count}")

        report_lines.append("\n  Detalhes das anomalias (primeiras 10):")
        for i, anomaly in enumerate(summary['anomalias'][:10], 1):
            report_lines.append(f"    {i}. Frame {anomaly['frame']}: {anomaly.get('description', 'Anomalia detectada')}")
    else:
        report_lines.append("  Nenhuma anomalia detectada.")

    report_lines.extend(["", "=" * 80, "Fim do Relatório", "=" * 80])
    report_text = "\n".join(report_lines)

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

    # Inicializa MediaPipe Pose
    pose, mp_pose, mp_drawing = _init_mediapipe_pose()

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

                # Converte frame para RGB para MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Detecta pose corporal
                pose_info = None
                pose_landmarks = None
                if pose is not None:
                    try:
                        pose_results = pose.process(rgb_frame)
                        if pose_results and pose_results.pose_landmarks:
                            pose_landmarks = pose_results.pose_landmarks.landmark
                            # Desenha landmarks de pose no frame (opcional)
                            if mp_drawing is not None and mp_pose is not None:
                                mp_drawing.draw_landmarks(
                                    frame,
                                    pose_results.pose_landmarks,
                                    mp_pose.POSE_CONNECTIONS
                                )
                            # Analisa pose para extrair informações
                            pose_info = analyze_pose_for_activity(pose_landmarks)
                    except Exception as e:
                        logger.debug(f"Erro ao processar pose: {e}")

                # Analisa o frame para obter dados das faces (filtra bonecos/ilustrações)
                faces_data = analyze_frame(frame, stats=stats)

                # Extrai posições das faces para análise de movimento
                current_positions = _extract_face_positions(faces_data)

                # Adiciona dados ao sistema de estatísticas e detecta atividades/anomalias
                frame_number = stats.frames_processed + 1
                _, anomalies = stats.add_frame_data(
                    frame_number,
                    faces_data,
                    previous_positions,
                    pose_info,
                    pose_landmarks
                )

                # Limpa atividades antigas da lista de visualização (otimização de memória)
                stats.active_activities = [
                    act for act in stats.active_activities
                    if frame_number - act['frame'] <= ACTIVITY_DISPLAY_DURATION
                ]

                # Processa o frame com todas as anotações (faces, emoções e anomalias)
                faces_in_frame, _ = process_single_frame(
                    frame,
                    display,
                    out,
                    anomalies=anomalies
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
        if pose is not None:
            try:
                pose.close()  # Libera recursos do MediaPipe Pose
            except Exception:
                pass  # Ignora erros ao fechar
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