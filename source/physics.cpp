#include "physics.h"
#include "physics_lad.h"
#include "physics_burgers.h"
#include "physics_ns.h"

std::unique_ptr<PhysicsModel> createPhysicsModel()
{
#if defined(LAD)
    return std::make_unique<PhysicsLAD>();
#elif defined(BURGERS)
    return std::make_unique<PhysicsBurgers>();
#elif defined(NS)
    return std::make_unique<PhysicsNS>();
#else
    static_assert(false, "No physics model defined. Define LAD, BURGERS, or NS.");
#endif
}
