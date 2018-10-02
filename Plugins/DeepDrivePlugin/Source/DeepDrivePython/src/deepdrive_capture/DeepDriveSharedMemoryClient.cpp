
#include "DeepDriveSharedMemoryClient.h"
#include "Public/SharedMemory/SharedMemory.h"

#include "Public/Messages/DeepDriveCaptureMessage.h"

#include "PyCaptureCameraObject.h"
#include "PyCaptureLastCollisionObject.h"
#include "PyCaptureSnapshotObject.h"

#include <iostream>
#include <string>

DeepDriveSharedMemoryClient::DeepDriveSharedMemoryClient()
	:	m_SharedMemory(new SharedMemory)
{
}

DeepDriveSharedMemoryClient::~DeepDriveSharedMemoryClient()
{
	delete m_SharedMemory;
}

bool DeepDriveSharedMemoryClient::connect(const std::string &name, uint32 maxSize)
{
	if(m_SharedMemory)
	{
		m_isConnected = m_SharedMemory->connect(FString(name), maxSize);
		if(m_isConnected)
		{
			m_maxSize = maxSize;
			std::cout << "Successfully connected to " << name << " with max size of " << m_maxSize << "\n";
		}
		else
		{
		//			std::cout << "Failed to connect to " << name << "\n";
		}
	}

	return m_isConnected;
}
	
PyCaptureSnapshotObject* DeepDriveSharedMemoryClient::readMessage()
{
	PyCaptureSnapshotObject* msg = 0;

	if(m_SharedMemory)
	{
		const DeepDriveCaptureMessage *captureMsg = reinterpret_cast<const DeepDriveCaptureMessage*> (m_SharedMemory->lockForReading(0));
		if (captureMsg)
		{
			if	(	captureMsg->message_id != 0
				&&	captureMsg->message_id != m_lastMsgId
				)
			{
//				std::cout << "Received capture message " << captureMsg->message_id << " message size " << captureMsg->message_size << " camera count " << captureMsg->num_cameras << " at " << captureMsg <<  "\n";
//				dumpSharedMemContent(captureMsg);
				if(captureMsg->message_type == DeepDriveMessageType::Capture)
				{
					msg = reinterpret_cast<PyCaptureSnapshotObject*> (PyCaptureSnapshotType.tp_new(&PyCaptureSnapshotType, 0, 0));

					if(msg)
					{
						msg->capture_timestamp = captureMsg->creation_timestamp;
						msg->sequence_number = captureMsg->sequence_number;
						msg->speed = captureMsg->speed;
						msg->is_game_driving = captureMsg->is_game_driving;
						msg->is_resetting = captureMsg->is_resetting;
						msg->camera_count = captureMsg->num_cameras;
						msg->distance_along_route = captureMsg->distance_along_route;
						msg->route_length = captureMsg->route_length;
						msg->distance_to_center_of_lane = captureMsg->distance_to_center_of_lane;
						msg->distance_to_next_agent = captureMsg->distance_to_next_agent;
						msg->distance_to_prev_agent = captureMsg->distance_to_prev_agent;
						msg->distance_to_next_opposing_agent = captureMsg->distance_to_next_opposing_agent;
						msg->is_passing = captureMsg->is_passing;
						msg->lap_number = captureMsg->lap_number;

						msg->steering = captureMsg->steering;
						msg->throttle = captureMsg->throttle;
						msg->brake = captureMsg->brake;
						msg->handbrake = captureMsg->handbrake;

						NumPyUtils::copyVector3(captureMsg->position, msg->position);
						NumPyUtils::copyVector3(captureMsg->rotation, msg->rotation);
						NumPyUtils::copyVector3(captureMsg->velocity, msg->velocity);
						NumPyUtils::copyVector3(captureMsg->acceleration, msg->acceleration);
						NumPyUtils::copyVector3(captureMsg->dimension, msg->dimension);
						NumPyUtils::copyVector3(captureMsg->angular_velocity, msg->angular_velocity);
						NumPyUtils::copyVector3(captureMsg->angular_acceleration, msg->angular_acceleration);
						NumPyUtils::copyVector3(captureMsg->forward_vector, msg->forward_vector);
						NumPyUtils::copyVector3(captureMsg->forward_vector, msg->forward_vector);
						NumPyUtils::copyVector3(captureMsg->up_vector, msg->up_vector);
						NumPyUtils::copyVector3(captureMsg->right_vector, msg->right_vector);

						setupLastCollision(captureMsg->last_collision, *msg->last_collision);

						if (captureMsg->num_cameras)
						{
							const DeepDriveCaptureCamera *ddCam = &captureMsg->cameras[0];

							PyObject *camList = PyList_New(captureMsg->num_cameras);

							uint32 offsetToNext = 0;
							uint32 curInd = 0;
							do
							{
								PyCaptureCameraObject *pyCam = buildCamera(*ddCam);
								PyList_SetItem(camList, curInd++, reinterpret_cast<PyObject*> (pyCam));

								offsetToNext = ddCam->offset_to_next_camera;
								ddCam = reinterpret_cast<const DeepDriveCaptureCamera*> (reinterpret_cast<const uint8*> (ddCam) + offsetToNext);
							} while (offsetToNext);

							msg->cameras = reinterpret_cast<PyListObject*> (camList);
						}
						else
						{
							// Retaining old cameras causes segfault
							msg->cameras = 0;
							msg->camera_count = 0;
//							std::cout << "No cameras\n";
						}
					}

				}
				else
				{
					std::cout << "Unknown message with type " << static_cast<uint32> (captureMsg->message_type) << " and Size " << captureMsg->message_size << " received\n";
				}
			}

			m_lastMsgId = captureMsg->message_id;
		}

		m_SharedMemory->unlock();

	}
	

	return msg;
}

void DeepDriveSharedMemoryClient::close()
{
	
}

PyCaptureCameraObject* DeepDriveSharedMemoryClient::buildCamera(const DeepDriveCaptureCamera &srcCam)
{
	PyCaptureCameraObject *dstCam = reinterpret_cast<PyCaptureCameraObject*> (PyCaptureCameraType.tp_new(&PyCaptureCameraType, 0, 0));

	if(dstCam)
	{
		dstCam->type = srcCam.type;
		dstCam->id = srcCam.id;
		dstCam->horizontal_field_of_view = srcCam.horizontal_field_of_view;
		dstCam->aspect_ratio = srcCam.aspect_ratio;

		dstCam->capture_width = srcCam.capture_width;
		dstCam->capture_height = srcCam.capture_height;
		dstCam->image_data = PyCaptureCameraObject::createImage(srcCam.capture_width * srcCam.capture_height * 3, srcCam.data);
		dstCam->depth_data = PyCaptureCameraObject::createImage(srcCam.capture_width * srcCam.capture_height, srcCam.data + srcCam.depth_offset);
	}

	return dstCam;
}

void DeepDriveSharedMemoryClient::setupLastCollision(const DeepDriveLastCollisionData &srcCollision, PyCaptureLastCollisionObject &dstCollision)
{
	dstCollision.time_utc = srcCollision.time_utc;
	dstCollision.time_stamp = srcCollision.time_stamp;
	dstCollision.time_since_last_collision = srcCollision.time_since_last_collision;

	const char *src = srcCollision.collision_location;
	char *dst = const_cast<char*> (dstCollision.collision_location);
	for (uint32 i = PyCaptureLastCollisionObject::StringSize - 1; i > 0 && *src; *dst++ = *src++, --i);
	*dst = 0;

	NumPyUtils::copyVector3(srcCollision.collidee_velocity, dstCollision.collidee_velocity);
	NumPyUtils::copyVector3(srcCollision.collision_normal, dstCollision.collision_normal);
}


void DeepDriveSharedMemoryClient::dumpSharedMemContent(const DeepDriveCaptureMessage *data)
{
	std::string name = "D:\\tmp\\aiworld\\shared_mem_dump_" + std::to_string(m_DumpIndex) + ".bin";

	FILE *out = fopen(name.c_str(), "wb");
	if(out)
	{
		FILE *outNum = fopen("D:\\tmp\\aiworld\\shared_mem_dump.txt", "w");
		if(outNum)
		{
			fprintf(outNum, "%d\n", m_DumpIndex);
			fclose(outNum);
		}

		fwrite(data, 1, m_maxSize, out);
		fclose(out);
		m_DumpIndex = (m_DumpIndex + 1) % 5;
	}
	std::cout << "Dumped\n";
}
