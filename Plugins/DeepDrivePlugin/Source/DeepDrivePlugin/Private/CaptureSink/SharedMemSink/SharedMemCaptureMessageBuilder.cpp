
#include "DeepDrivePluginPrivatePCH.h"
#include "Private/CaptureSink/SharedMemSink/SharedMemCaptureMessageBuilder.h"
#include "Private/Capture/CaptureBuffer.h"

#include "Public/Messages/DeepDriveCaptureMessage.h"
#include "Public/SharedMemory/SharedMemory.h"
#include "Public/DeepDriveData.h"

#include "Private/Utils/DeepDriveUtils.h"

DEFINE_LOG_CATEGORY(LogSharedMemCaptureMessageBuilder);

SharedMemCaptureMessageBuilder::SharedMemCaptureMessageBuilder(SharedMemory &sharedMem, uint8 *buffer)
:	m_SharedMem(sharedMem)
{
	m_MessageBuffer = buffer;
}

void SharedMemCaptureMessageBuilder::begin(const DeepDriveDataOut &deepDriveData, double timestamp, uint32 sequenceNumber)
{
	//m_Message = reinterpret_cast<DeepDriveCaptureMessage*> (m_SharedMem.lockForWriting(0));
	m_Message = reinterpret_cast<DeepDriveCaptureMessage*> (m_MessageBuffer);

	if(m_Message)
	{
		m_Message = new (m_Message) DeepDriveCaptureMessage();

		m_Message->sequence_number = sequenceNumber;
		m_Message->creation_timestamp = timestamp;
		m_Message->padding_0 = 0xEFBEADDE;

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
		m_Message->scenario_finished = 0;
		m_Message->num_cameras = 0;
		m_Message->distance_along_route = deepDriveData.DistanceAlongRoute;
		m_Message->route_length = deepDriveData.RouteLength;
		m_Message->distance_to_center_of_lane = deepDriveData.DistanceToCenterOfLane;
		m_Message->distance_to_next_agent = deepDriveData.DistanceToNextAgent;
		m_Message->distance_to_prev_agent = deepDriveData.DistanceToPrevAgent;
		m_Message->distance_to_next_opposing_agent = deepDriveData.DistanceToNextOpposingAgent;
		m_Message->is_passing = deepDriveData.IsPassing ? 1 : 0;
		m_Message->lap_number = deepDriveData.LapNumber;

		m_Message->last_collision.time_utc = deepDriveData.CollisionData.LastCollisionTimeUTC.ToUnixTimestamp();
		m_Message->last_collision.time_stamp = deepDriveData.CollisionData.LastCollisionTimeStamp;
		m_Message->last_collision.time_since_last_collision = deepDriveData.CollisionData.TimeSinceLastCollision;
		m_Message->last_collision.collidee_velocity = DeepDriveVector3(deepDriveData.CollisionData.ColliderVelocity);
		m_Message->last_collision.collision_normal = DeepDriveVector3(deepDriveData.CollisionData.CollisionNormal);
		deepdrive::utils::copyString(deepDriveData.CollisionData.CollisionLocation, m_Message->last_collision.collision_location, DeepDriveMessageHeader::StringSize);

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

		FFloat16 *colDst = reinterpret_cast<FFloat16*>( &curCamera->data[0] );
		FFloat16 *depthDst = colDst + width * height * 3;

		CaptureBuffer *secondaryCaptureBuffer = captureBuffer.getSecondaryCaptureBuffer();

		EDeepDriveInternalCaptureEncoding encoding = captureBuffer.getEncoding();
		switch(encoding)
		{
			case EDeepDriveInternalCaptureEncoding::RGB_DEPTH:
				decodeRGBDepth(captureBuffer, colDst, depthDst);
				break;
			case EDeepDriveInternalCaptureEncoding::SEPARATE:
				if(secondaryCaptureBuffer)
					decodeSeparate(captureBuffer, *secondaryCaptureBuffer, colDst, depthDst);
				break;
			case EDeepDriveInternalCaptureEncoding::GRAYSCALE_DEPTH:
				decodeGrayscaleDepth(captureBuffer, colDst, depthDst);
				break;
			case EDeepDriveInternalCaptureEncoding::COMPRESSED_YUV_DEPTH:
				{
					decodeCompressedYUVDepth(captureBuffer, colDst, depthDst);
				}
				break;
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
	DeepDriveCaptureMessage *finalMsgBuf = reinterpret_cast<DeepDriveCaptureMessage*> (m_SharedMem.lockForWriting(0));

	if(finalMsgBuf)
	{
		FMemory::BigBlockMemcpy(finalMsgBuf, m_Message, m_MessageSize);
		finalMsgBuf->message_size = m_MessageSize;
		finalMsgBuf->setMessageId();
		m_SharedMem.unlock(m_MessageSize);
//		UE_LOG(LogSharedMemCaptureMessageBuilder, Log, TEXT("SharedMemCaptureMessageBuilder::flush Flushed message %d msgSize %d"), m_Message->message_id, m_MessageSize);
	}
}

void SharedMemCaptureMessageBuilder::decodeGrayscaleDepth(CaptureBuffer &captureBuffer, FFloat16 *colDst, FFloat16 *depthDst)
{
	const uint32 width = captureBuffer.getWidth();
	const uint32 height = captureBuffer.getHeight();
	const FFloat16 *f16ColSrc = captureBuffer.getBuffer<FFloat16>();
	for(unsigned y = 0; y < height; y++)
	{
		uint32 ind = 0;
		for(unsigned x = 0; x < width; x++)
		{
			const FFloat16 f16 = f16ColSrc[ind];
			*colDst++ = f16;
			*colDst++ = f16;
			*colDst++ = f16;

			depthDst->SetWithoutBoundsChecks(f16ColSrc[ind + 1].GetFloat() / 65535.0f);
			depthDst++;

			ind += 4;
		}

		f16ColSrc = reinterpret_cast<const FFloat16*> (reinterpret_cast<const uint8*> (f16ColSrc) + captureBuffer.getStride() );
	}
}

void SharedMemCaptureMessageBuilder::decodeCompressedYUVDepth(CaptureBuffer &captureBuffer, FFloat16 *colDst, FFloat16 *depthDst)
{
	const uint32 width = captureBuffer.getWidth();
	const uint32 height = captureBuffer.getHeight();
	const FFloat16 *f16ColSrc = captureBuffer.getBuffer<FFloat16>();
	for(unsigned y = 0; y < height; y++)
	{
		uint32 ind = 0;
		for(unsigned x = 0; x < width; x++)
		{
			const float Y = f16ColSrc[ind].GetFloat();
			const FFloat16 UV = f16ColSrc[ind + 1].GetFloat();
			const float v = FGenericPlatformMath::Frac(UV);
			const float u = 0.5f * (UV - v);
			const FVector yuv(Y, u, v);

			colDst->SetWithoutBoundsChecks( FVector::DotProduct(yuv, FVector(1.0f, 0.0f, 1.3983f)) );			colDst++;
			colDst->SetWithoutBoundsChecks( FVector::DotProduct(yuv, FVector(1.0f, -0.21482f, -0.38059f)) );	colDst++;
			colDst->SetWithoutBoundsChecks( FVector::DotProduct(yuv, FVector(1.0f, 2.12798f, 0.0f)) );			colDst++;

			depthDst->SetWithoutBoundsChecks(f16ColSrc[ind + 2].GetFloat() / 65535.0f);
			depthDst++;

			ind += 4;
		}

		f16ColSrc = reinterpret_cast<const FFloat16*> (reinterpret_cast<const uint8*> (f16ColSrc) + captureBuffer.getStride() );
	}
}

void SharedMemCaptureMessageBuilder::decodeRGBDepth(CaptureBuffer &captureBuffer, FFloat16 *colDst, FFloat16 *depthDst)
{
	const uint32 width = captureBuffer.getWidth();
	const uint32 height = captureBuffer.getHeight();
	const FFloat16 *f16ColSrc = captureBuffer.getBuffer<FFloat16>();
	for(unsigned y = 0; y < height; y++)
	{
		uint32 ind = 0;
		for(unsigned x = 0; x < width; x++)
		{
			*colDst++ = f16ColSrc[ind++];
			*colDst++ = f16ColSrc[ind++];
			*colDst++ = f16ColSrc[ind++];

			depthDst->SetWithoutBoundsChecks(f16ColSrc[ind++].GetFloat() / 65535.0f);
			depthDst++;
		}

		f16ColSrc = reinterpret_cast<const FFloat16*> (reinterpret_cast<const uint8*> (f16ColSrc) + captureBuffer.getStride() );
	}
}

void SharedMemCaptureMessageBuilder::decodeSeparate(CaptureBuffer &sceneCaptureBuffer, CaptureBuffer &depthCaptureBuffer, FFloat16 *colDst, FFloat16 *depthDst)
{
	const uint32 width = sceneCaptureBuffer.getWidth();
	const uint32 height = sceneCaptureBuffer.getHeight();
	const FFloat16 *f16ColSrc = sceneCaptureBuffer.getBuffer<FFloat16>();
	const FFloat16 *f16DepthSrc = depthCaptureBuffer.getBuffer<FFloat16>();
	
	for(unsigned y = 0; y < height; y++)
	{
		uint32 ind = 0;
		for(unsigned x = 0; x < width; x++)
		{
			*colDst++ = f16ColSrc[ind++];
			*colDst++ = f16ColSrc[ind++];
			*colDst++ = f16ColSrc[ind++];

			// skip unused alpha in scene source buffer
			++ind;
			depthDst->SetWithoutBoundsChecks(f16DepthSrc[x].GetFloat() / 65535.0f);
			depthDst++;
		}

		f16ColSrc = reinterpret_cast<const FFloat16*> (reinterpret_cast<const uint8*> (f16ColSrc) + sceneCaptureBuffer.getStride() );
		f16DepthSrc = reinterpret_cast<const FFloat16*> (reinterpret_cast<const uint8*> (f16DepthSrc) + depthCaptureBuffer.getStride() );
	}
}
