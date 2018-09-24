import unreal_engine as ue
import sys
print(sys.executable)
print(sys.exc_info())
import deepdrive_client

# TODO: Automate setting PYTHONHOME in DefaultEngine.ini in deepdrive repo
# TODO: Set the ScriptsDir in C# correctly in deepdrive repo

#
#
# worlds = ue.all_worlds()
#
# sim_world = [w for w in worlds if 'DeepDriveSim_Demo.DeepDriveSim_Demo' in w.get_full_name()][-1]
#
# print(sim_world.get_full_name())
#
# controllers = [(a.get_full_name(), a)
#                for a in sim_world.all_actors() if 'localaicontroller_' in a.get_full_name().lower()]
# print(controllers)
# controller = controllers[-1][1]
# print(dir(controller))
# print(controller.functions())
# print(controller.getIsPassing())

# [2018.09.23-01.02.08:998][652]LogPython: ['Configure', 'UnPossess', 'StopMovement', 'SetInitialLocationAndRotation', 'SetIgnoreMoveInput', 'SetIgnoreLookInput', 'SetControlRotation', 'ResetIgnoreMoveInput', 'ResetIgnoreLookInput', 'ResetIgnoreInputFlags', 'ReceiveInstigatedAnyDamage', 'Possess', 'OnRep_PlayerState', 'OnRep_Pawn', 'LineOfSightTo', 'K2_GetPawn', 'IsPlayerController', 'IsMoveInputIgnored', 'IsLookInputIgnored', 'IsLocalPlayerController', 'IsLocalController', 'GetViewTarget', 'GetDesiredRotation', 'GetControlRotation', 'ClientSetRotation', 'ClientSetLocation', 'CastToPlayerController', 'WasRecentlyRendered', 'UserConstructionScript', 'TearOff', 'SnapRootComponentTo', 'SetTickGroup', 'SetTickableWhenPaused', 'SetReplicates', 'SetReplicateMovement', 'SetOwner', 'SetNetDormancy', 'SetLifeSpan', 'SetIsTemporarilyHiddenInEditor', 'SetActorTickInterval', 'SetActorTickEnabled', 'SetActorScale3D', 'SetActorRelativeScale3D', 'SetActorHiddenInGame', 'SetActorEnableCollision', 'RemoveTickPrerequisiteComponent', 'RemoveTickPrerequisiteActor', 'ReceiveTick', 'ReceiveRadialDamage', 'ReceivePointDamage', 'ReceiveHit', 'ReceiveEndPlay', 'ReceiveDestroyed', 'ReceiveBeginPlay', 'ReceiveAnyDamage', 'ReceiveActorOnReleased', 'ReceiveActorOnInputTouchLeave', 'ReceiveActorOnInputTouchEnter', 'ReceiveActorOnInputTouchEnd', 'ReceiveActorOnInputTouchBegin', 'ReceiveActorOnClicked', 'ReceiveActorEndOverlap', 'ReceiveActorEndCursorOver', 'ReceiveActorBeginOverlap', 'ReceiveActorBeginCursorOver', 'PrestreamTextures', 'OnRep_ReplicateMovement', 'OnRep_ReplicatedMovement', 'OnRep_Owner', 'OnRep_Instigator', 'OnRep_AttachmentReplication', 'MakeNoise', 'MakeMIDForMaterial', 'K2_TeleportTo', 'K2_SetActorTransform', 'K2_SetActorRotation', 'K2_SetActorRelativeTransform', 'K2_SetActorRelativeRotation', 'K2_SetActorRelativeLocation', 'K2_SetActorLocationAndRotation', 'K2_SetActorLocation', 'K2_OnReset', 'K2_OnEndViewTarget', 'K2_OnBecomeViewTarget', 'K2_GetRootComponent', 'K2_GetActorRotation', 'K2_GetActorLocation', 'K2_DetachFromActor', 'K2_DestroyComponent', 'K2_DestroyActor', 'K2_AttachToComponent', 'K2_AttachToActor', 'K2_AttachRootComponentToActor', 'K2_AttachRootComponentTo', 'K2_AddActorWorldTransform', 'K2_AddActorWorldRotation', 'K2_AddActorWorldOffset', 'K2_AddActorLocalTransform', 'K2_AddActorLocalRotation', 'K2_AddActorLocalOffset', 'IsTemporarilyHiddenInEditor', 'IsSelectable', 'IsOverlappingActor', 'IsHiddenEdAtStartup', 'IsHiddenEd', 'IsEditable', 'IsChildActor', 'IsActorTickEnabled', 'IsActorBeingDestroyed', 'HasAuthority', 'GetVerticalDistanceTo', 'GetVelocity', 'GetTransform', 'GetTickableWhenPaused', 'GetSquaredDistanceTo', 'GetRemoteRole', 'GetParentComponent', 'GetParentActor', 'GetOwner', 'GetOverlappingComponents', 'GetOverlappingActors', 'GetLifeSpan', 'GetInstigatorController', 'GetInstigator', 'GetInputVectorAxisValue', 'GetInputAxisValue', 'GetInputAxisKeyValue', 'GetHorizontalDotProductTo', 'GetHorizontalDistanceTo', 'GetGameTimeSinceCreation', 'GetDotProductTo', 'GetDistanceTo', 'GetComponentsByTag', 'GetComponentsByClass', 'GetComponentByClass', 'GetAttachParentSocketName', 'GetAttachParentActor', 'GetAttachedActors', 'GetAllChildActors', 'GetActorUpVector', 'GetActorTimeDilation', 'GetActorTickInterval', 'GetActorScale3D', 'GetActorRightVector', 'GetActorRelativeScale3D', 'GetActorForwardVector', 'GetActorEyesViewPoint', 'GetActorEnableCollision', 'GetActorBounds', 'ForceNetUpdate', 'FlushNetDormancy', 'EnableInput', 'DisableInput', 'DetachRootComponentFromParent', 'AddTickPrerequisiteComponent', 'AddTickPrerequisiteActor', 'AddComponent', 'ActorHasTag', 'ExecuteUbergraph']

# ['Configure', 'UnPossess', 'StopMovement', 'SetInitialLocationAndRotation', 'SetIgnoreMoveInput', 'SetIgnoreLookInput', 'SetControlRotation', 'ResetIgnoreMoveInput', 'ResetIgnoreLookInput', 'ResetIgnoreInputFlags', 'ReceiveInstigatedAnyDamage', 'Possess', 'OnRep_PlayerState', 'OnRep_Pawn', 'LineOfSightTo', 'K2_GetPawn', 'IsPlayerController', 'IsMoveInputIgnored', 'IsLookInputIgnored', 'IsLocalPlayerController', 'IsLocalController', 'GetViewTarget', 'GetDesiredRotation', 'GetControlRotation', 'ClientSetRotation', 'ClientSetLocation', 'CastToPlayerController', 'WasRecentlyRendered', 'UserConstructionScript', 'TearOff', 'SnapRootComponentTo', 'SetTickGroup', 'SetTickableWhenPaused', 'SetReplicates', 'SetReplicateMovement', 'SetOwner', 'SetNetDormancy', 'SetLifeSpan', 'SetIsTemporarilyHiddenInEditor', 'SetActorTickInterval', 'SetActorTickEnabled', 'SetActorScale3D', 'SetActorRelativeScale3D', 'SetActorHiddenInGame', 'SetActorEnableCollision', 'RemoveTickPrerequisiteComponent', 'RemoveTickPrerequisiteActor', 'ReceiveTick', 'ReceiveRadialDamage', 'ReceivePointDamage', 'ReceiveHit', 'ReceiveEndPlay', 'ReceiveDestroyed', 'ReceiveBeginPlay', 'ReceiveAnyDamage', 'ReceiveActorOnReleased', 'ReceiveActorOnInputTouchLeave', 'ReceiveActorOnInputTouchEnter', 'ReceiveActorOnInputTouchEnd', 'ReceiveActorOnInputTouchBegin', 'ReceiveActorOnClicked', 'ReceiveActorEndOverlap', 'ReceiveActorEndCursorOver', 'ReceiveActorBeginOverlap', 'ReceiveActorBeginCursorOver', 'PrestreamTextures', 'OnRep_ReplicateMovement', 'OnRep_ReplicatedMovement', 'OnRep_Owner', 'OnRep_Instigator', 'OnRep_AttachmentReplication', 'MakeNoise', 'MakeMIDForMaterial', 'K2_TeleportTo', 'K2_SetActorTransform', 'K2_SetActorRotation', 'K2_SetActorRelativeTransform', 'K2_SetActorRelativeRotation', 'K2_SetActorRelativeLocation', 'K2_SetActorLocationAndRotation', 'K2_SetActorLocation', 'K2_OnReset', 'K2_OnEndViewTarget', 'K2_OnBecomeViewTarget', 'K2_GetRootComponent', 'K2_GetActorRotation', 'K2_GetActorLocation', 'K2_DetachFromActor', 'K2_DestroyComponent', 'K2_DestroyActor', 'K2_AttachToComponent', 'K2_AttachToActor', 'K2_AttachRootComponentToActor', 'K2_AttachRootComponentTo', 'K2_AddActorWorldTransform', 'K2_AddActorWorldRotation', 'K2_AddActorWorldOffset', 'K2_AddActorLocalTransform', 'K2_AddActorLocalRotation', 'K2_AddActorLocalOffset', 'IsTemporarilyHiddenInEditor', 'IsSelectable', 'IsOverlappingActor', 'IsHiddenEdAtStartup', 'IsHiddenEd', 'IsEditable', 'IsChildActor', 'IsActorTickEnabled', 'IsActorBeingDestroyed', 'HasAuthority', 'GetVerticalDistanceTo', 'GetVelocity', 'GetTransform', 'GetTickableWhenPaused', 'GetSquaredDistanceTo', 'GetRemoteRole', 'GetParentComponent', 'GetParentActor', 'GetOwner', 'GetOverlappingComponents', 'GetOverlappingActors', 'GetLifeSpan', 'GetInstigatorController', 'GetInstigator', 'GetInputVectorAxisValue', 'GetInputAxisValue', 'GetInputAxisKeyValue', 'GetHorizontalDotProductTo', 'GetHorizontalDistanceTo', 'GetGameTimeSinceCreation', 'GetDotProductTo', 'GetDistanceTo', 'GetComponentsByTag', 'GetComponentsByClass', 'GetComponentByClass', 'GetAttachParentSocketName', 'GetAttachParentActor', 'GetAttachedActors', 'GetAllChildActors', 'GetActorUpVector', 'GetActorTimeDilation', 'GetActorTickInterval', 'GetActorScale3D', 'GetActorRightVector', 'GetActorRelativeScale3D', 'GetActorForwardVector', 'GetActorEyesViewPoint', 'GetActorEnableCollision', 'GetActorBounds', 'ForceNetUpdate', 'FlushNetDormancy', 'EnableInput', 'DisableInput', 'DetachRootComponentFromParent', 'AddTickPrerequisiteComponent', 'AddTickPrerequisiteActor', 'AddComponent', 'ActorHasTag', 'ExecuteUbergraph']
# [('DeepDriveInputController_C /Game/DeepDrive/Maps/UEDPIE_0_DeepDriveSim_Demo.DeepDriveSim_Demo:PersistentLevel.DeepDriveInputController_C_0', <unreal_engine.UObject object at 0x7fa5febe5d20>), ('DeepDriveAgentLocalAIController /Game/DeepDrive/Maps/UEDPIE_0_DeepDriveSim_Demo.DeepDriveSim_Demo:PersistentLevel.DeepDriveAgentLocalAIController_0', <unreal_engine.UObject object at 0x7fa5febe5d50>), ('DeepDriveAgentLocalAIController /Game/DeepDrive/Maps/UEDPIE_0_DeepDriveSim_Demo.DeepDriveSim_Demo:PersistentLevel.DeepDriveAgentLocalAIController_1', <unreal_engine.UObject object at 0x7fa5febe5d80>), ('LocalAIControllerCreator_C /Game/DeepDrive/Maps/UEDPIE_0_DeepDriveSim_Demo.DeepDriveSim_Demo:PersistentLevel.LocalAIControllerCreator_76', <unreal_engine.UObject object at 0x7fa5febee690>), ('ManualControllerCreator_C /Game/DeepDrive/Maps/UEDPIE_0_DeepDriveSim_Demo.DeepDriveSim_Demo:PersistentLevel.ManualControllerCreator_68', <unreal_engine.UObject object at 0x7fa5febee6c0>), ('RemoteAIControllerCreator_C /Game/DeepDrive/Maps/UEDPIE_0_DeepDriveSim_Demo.DeepDriveSim_Demo:PersistentLevel.RemoteAIControllerCreator_84', <unreal_engine.UObject object at 0x7fa5febee6f0>)]

# local_ai_controller = controllers[1][1]

# from unreal_engine.classes import DeepDriveAgentLocalAIController

# print(dir(ue.all_worlds()[0]))

# print(local_ai_controller.get_class())
# print(local_ai_controller.__class__)
# print(local_ai_controller.functions())
# print(local_ai_controller.GetActorBounds())
#
# # print(ue.all_worlds()[0].get_components_by_class(DeepDriveAgentLocalAIController))
#
#
# agents = [(a.get_full_name(), a)
#                for a in ue.all_worlds()[0].all_actors() if 'agent' in a.get_full_name().lower()]
#
# print(agents)
#
# print(agents[0][1].ChaseCamera)

# passing_states = [(a.get_full_name(), a)
#                for a in ue.all_worlds()[0].all_actors() if 'aistate' in a.get_full_name().lower()]

# print(passing_states)