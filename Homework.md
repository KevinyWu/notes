# Homework

# Openflow

Explain Figure 2 in the paper and how did OpenFlow build upon commercial routers and switches without inventing a new type of programmable hardware for a router or switch.

- Figure 2 shows regular commercial switches augmented with a Flow Table and Secure Channel, letting a remote Controller direct traffic while keeping the original hardware untouched
- OpenFlow reused flow-tables already built into commercial switches, adding a standard way to program them remotely
- Experimental and production traffic were kept separate using either VLANs or by sending normal packets through the switch's regular pipeline

Pick two examples discussed in Section 3 and outline in your own words how these examples were built on top of OpenFlow. What was easy and what was hard?

- VLANs
	- OpenFlow lets users have their own isolated network by statically declaring flows that specify which ports are accessible for a given VLAN ID, with switches tagging traffic from specific users with the appropriate VLAN ID
	- Easy: the static approach mapped naturally onto OpenFlow, since declaring flows and tagging traffic with VLAN IDs is straightforward with flow table rules
	- Hard: the more dynamic approach adds complexity on top of OpenFlow
- Non-IP Network
	- OpenFlow allows experiments with entirely new naming, addressing and routing schemes by matching packets based on their header alone, without requiring IP formatting
	- Easy: OpenFlow already supports matching on Ethernet headers like MAC source and destination addresses
	- Hard: the more general case of letting a controller define a custom mask (offset + value + mask)

# Ethane

"This paper presents Ethane, a new network architecture for the enterprise. Ethane allows managers to define a single networkwide fine-grain policy, and then enforces it directly. Ethane couples extremely simple flow-based Ethernet switches with a centralized controller that manages the admittance and routing of flows. While radical, this design is backwards-compatible with existing hosts and switches"

Explain the steps involved in setting up and tearing down a flow in Ethane including how does the controller interact with the switches in the process. (Section 2,2, 3.2, 3.3)

- When a new packet arrives with no matching flow entry, the switch forwards it to the controller along with the physical port it arrived on
- The controller checks the policy to allow, deny, or route through a waypoint; if allowed, it computes the route and installs flow entries in every switch along the path
- Flows are torn down either by timing out due to inactivity, or by the controller explicitly revoking them when a host leaves or its privileges change

Explain what is unique about Pol-Eth policy language. How does Ethane convert specifications written in Pol-Eth to flow level actions in Ethane? (Section 4)

- Policies are declared over high-level names like users and hosts rather than low-level addresses
- Rules consist of conditions and actions, where actions include allow, deny, or routing through a waypoint
- Pol-Eth is compiled rather than interpreted since decisions must be made very fast
- The first time a sender talks to a new receiver, a custom permission check function is dynamically created for that pair
- If allowed, the controller installs the corresponding flow entries into the switches along the path

# Netbricks

How is notion of network functions (both traditional NFVs and NFs without the V) as discussed in NetBricks differ from conventional programmable routers in SDNs? How is this abstraction deeper in the specification space?

- SDN switches forward packets based on flow table rules, while NetBricks NFs implement complex processing like firewalls, NATs, and load balancers
- SDN matches packet headers, while NetBricks operates deeper, transforming payloads, managing TCP bytestreams, and chaining multiple processing steps using abstractions
- Traditional NFV isolated these complex NFs using VMs or containers, causing huge performance penalties. NetBricks replaces hardware isolation with software isolation using Rust

How does NetBricks support Network Functions without virtualization?

Explain this argument mention below in your own words and provide supporting evidence for this argument from the evaluation in Section 5:

"NetBricks provides both a programming model (for building NFs) and an execution environment (for running NFs). The programming model is built around a core set of high-level but customizable abstractions for common packet processing tasks; to demonstrate the generality of these abstractions and the efficiency of their implementations, we reimplemented 5 existing NFs in NetBricks and show that they perform well compared to their native versions. Our execution environment relies on the use of safe languages and runtimes for memory and fault isolation (similar to existing systems we rely on scheduling for performance isolation). Inter-process communication is also important in NF deployments, and IPC in these deployments must ensure that messages cannot be modified by an NF after being sent, a property we refer to as packet isolation. Current systems copy packets to ensure packet isolation, we instead use static check to provide this property without copies. The resulting design, which we call Zero-Copy Software Isolation (ZCSI), is the first to achieve memory and packet isolation with no performance penalty (in stark contrast to virtualization)."

- Instead of VMs or containers, NetBricks uses Rust to enforce memory isolation through compile-time checks, no hardware boundaries needed
- Packets are passed between NFs via function calls instead of being copied, with the language statically ensuring only one NF touches a packet at a time (ZCSI) (End of section 2.2)
- NetBricks matched native C performance for simple NFs (23.2 vs 23.3 Mpps) and outperformed original implementations, e.g. 3x better NAT throughput (Section 5.2.1)
- A single NF in a container was 2.7x slower than NetBricks; in a VM even worse (Table 1)
- For chained NFs, NetBricks was up to 7x faster than containers and 11x faster than VMs (Figure 8)
