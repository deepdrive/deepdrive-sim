
#include "DeepDrivePluginPrivatePCH.h"
#include "Private/CaptureSink/SharedMemSink/SharedMemCaptureMessageBuilder.h"

#include "Public/Messages/DeepDriveCaptureMessage.h"
#include "Public/SharedMemory/SharedMemory.h"


DEFINE_LOG_CATEGORY(LogSharedMemCaptureMessageBuilder);

SharedMemCaptureMessageBuilder::SharedMemCaptureMessageBuilder(SharedMemory &sharedMem)
	:	m_SharedMem(sharedMem)
{
}

void SharedMemCaptureMessageBuilder::begin(const FDeepDriveDataOut &deepDriveData, double timestamp, uint32 sequenceNumber)
{
	m_Message = reinterpret_cast<DeepDriveCaptureMessage*> (m_SharedMem.lockForWriting(0));

	if(m_Message)
	{

		m_Message = new (m_Message) DeepDriveCaptureMessage();

		m_Message->sequence_number = sequenceNumber;
		m_Message->creation_timestamp = timestamp;
		m_Message->padding_0 = 0xEFBEADDE;
		m_Message->padding_1 = 0xEFBEADDE;

		m_Message->position = DeepDriveVector3(deepDriveData.Position);

		FVector euler = deepDriveData.Rotation.Euler();
		m_Message->rotation = DeepDriveVector3( FMath::DegreesToRadians(euler.X), FMath::DegreesToRadians(euler.Y), FMath::DegreesToRadians(euler.Z) );
		
		m_Message->velocity = DeepDriveVector3(deepDriveData.Velocity);
		m_Message->acceleration = DeepDriveVector3(deepDriveData.Acceleration);
		m_Message->angular_velocity = DeepDriveVector3(deepDriveData.AngularVelocity);
		m_Message->angular_acceleration = DeepDriveVector3(deepDriveData.AngularAcceleration);

		FQuat quat = deepDriveData.Rotation.Quaternion();
		m_Message->forward_vector = DeepDriveVector3(quat.GetForwardVector());
		m_Message->up_vector = DeepDriveVector3(quat.GetForwardVector());
		m_Message->right_vector = DeepDriveVector3(quat.GetForwardVector());

		m_Message->dimension = DeepDriveVector3(deepDriveData.Dimension);
		m_Message->speed = deepDriveData.Speed;
		m_Message->steering = deepDriveData.Steering;
		m_Message->throttle = deepDriveData.Throttle;
		m_Message->brake = deepDriveData.Brake;
		m_Message->handbrake = deepDriveData.Handbrake;
		m_Message->is_game_driving = deepDriveData.IsGameDriving;
		m_Message->is_resetting = deepDriveData.IsResetting;
		m_Message->num_cameras = 0;
		m_Message->distance_along_route = deepDriveData.DistanceAlongRoute;
		m_Message->distance_to_center_of_lane = deepDriveData.DistanceToCenterOfLane;
		m_Message->lap_number = deepDriveData.LapNumber;


		m_MessageSize = sizeof(DeepDriveCaptureMessage);
		m_remainingSize = m_SharedMem.getMaxPayloadSize() - sizeof(DeepDriveCaptureMessage);
		m_nextCamera = m_Message->cameras;
		m_prevCamera = 0;
		m_prevCameraSize = 0;

		//UE_LOG(LogSharedMemCaptureMessageBuilder, Log, TEXT("SharedMemCaptureMessageBuilder::begin %p %p | %p"), &m_Message->num_cameras, &m_Message->cameras, &m_nextCamera->type);
	}
	else
	{
		UE_LOG(LogSharedMemCaptureMessageBuilder, Error, TEXT("SharedMemCaptureMessageBuilder::begin Couldn't lock shared mem for writing") );
	}
}

void SharedMemCaptureMessageBuilder::addCamera(EDeepDriveCameraType camType, int32 camId, CaptureBuffer &captureBuffer)
{
	uint32 estimatedSize = sizeof(DeepDriveCaptureCamera);
	const uint32 width = captureBuffer.getWidth();
	const uint32 height = captureBuffer.getHeight();
	const CaptureBuffer::DataType dataType = captureBuffer.getDataType();

	estimatedSize += width * height * 3 * 2;		// 2 bytes per pixel rgb color buffer
	estimatedSize += width * height * 2;			// 2 bytes per depth value

	if	(	static_cast<int32> (estimatedSize) < m_remainingSize
		&&	dataType == CaptureBuffer::Float16
		)
	{
		DeepDriveCaptureCamera *curCamera = m_nextCamera;

		curCamera->type = static_cast<uint32> (camType);
		curCamera->id = camId;
		curCamera->offset_to_next_camera = 0;
		curCamera->horizontal_field_of_view = 1.7654;
		curCamera->aspect_ratio	= 1.0;
		curCamera->capture_width = width;
		curCamera->capture_height = height;
		curCamera->bytes_per_pixel = 6;
		curCamera->bytes_per_depth_value = 2;
		curCamera->depth_offset = width * height * curCamera->bytes_per_pixel;

		const FFloat16 *f16Src = captureBuffer.getBuffer<FFloat16>();
		FFloat16 *colDst = reinterpret_cast<FFloat16*>( &curCamera->data[0] );
		FFloat16 *depthDst = colDst + width * height * 3;

		for(unsigned y = 0; y < height; y++)
		{
			uint32 ind = 0;
			for(unsigned x = 0; x < width; x++)
			{
				*colDst++ = f16Src[ind++];
				*colDst++ = f16Src[ind++];
				*colDst++ = f16Src[ind++];

				depthDst->Set(f16Src[ind++].GetFloat() / 65535.0f);
				depthDst++;
			}

			f16Src = reinterpret_cast<const FFloat16*> (reinterpret_cast<const uint8*> (f16Src) + captureBuffer.getStride() );
		}

		// calc size for this camera
		const uint32 camMemSize = width * height * (curCamera->bytes_per_pixel + curCamera->bytes_per_depth_value) + sizeof(DeepDriveCaptureCamera);
		m_MessageSize += camMemSize;
		m_remainingSize -= camMemSize;

		// correct previous camera
		if(m_prevCamera)
		{
			m_prevCamera->offset_to_next_camera = m_prevCameraSize;
		}

		m_prevCamera = curCamera;
		m_prevCameraSize = camMemSize;

		m_nextCamera = reinterpret_cast<DeepDriveCaptureCamera*> (reinterpret_cast<uint8*> (curCamera) + camMemSize );

		++m_Message->num_cameras;
	}

}


void SharedMemCaptureMessageBuilder::flush()
{
	if(m_Message)
	{
		m_Message->message_size = m_MessageSize;
		m_Message->setMessageId();

#if 0
		FILE *out = fopen("D:\\tmp\\sm_dump_Unreal.bin", "wb");
		if (out)
		{
			fwrite(m_Message, sizeof(uint8), 512, out);
			fclose(out);
		}
#endif

		m_SharedMem.unlock(m_MessageSize);
//		UE_LOG(LogSharedMemCaptureMessageBuilder, Log, TEXT("SharedMemCaptureMessageBuilder::flush Flushed message %d msgSize %d"), m_Message->message_id, m_MessageSize);
	}
}
